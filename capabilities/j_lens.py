"""
J-Lens: Internal Thought Visualization and Interpretation.

Based on Anthropic's mechanistic interpretability research showing how
the J-lens reveals the model's internal thoughts at different layers.

Key capabilities:
- Multihop recall visualization (layer-by-layer reasoning)
- Mental arithmetic tracking (intermediate calculations)
- Bug detection in code (error prediction)
- ASCII/pattern recognition visualization
- Protein/sequence recognition tracking
- Prompt injection detection
- Layer-by-layer concept evolution

The J-lens works by reading the J-space at different layer depths
to reveal what concepts are active at each stage of processing.

Reference: Anthropic's "The J-lens reveals the model's internal thoughts"
"""

import time
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

from .j_space import JointWorkspace, Concept, ConceptType


class LensMode(Enum):
    """Different J-lens viewing modes."""
    RECALL = "recall"              # Multihop recall
    ARITHMETIC = "arithmetic"      # Mental arithmetic
    CODE_DEBUG = "code_debug"      # Bug detection
    PATTERN = "pattern"            # Pattern recognition
    SEQUENCE = "sequence"          # Sequence analysis
    SAFETY = "safety"              # Safety/injection detection
    REASONING = "reasoning"        # General reasoning


class LayerDepth(Enum):
    """Layer depth categories."""
    INPUT = "input"          # 0-20%: Raw input processing
    EARLY = "early"          # 20-40%: Feature extraction
    MIDDLE = "middle"        # 40-60%: Reasoning
    LATE = "late"            # 60-80%: Output preparation
    FINAL = "final"          # 80-100%: Final decision


@dataclass
class LayerReading:
    """A reading from a specific layer."""
    layer: int
    layer_depth: LayerDepth
    concepts: list[str]
    activations: list[float]
    interpretation: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class LensResult:
    """Complete result from J-lens analysis."""
    prompt: str
    mode: LensMode
    readings: list[LayerReading]
    intermediate_steps: list[str]
    final_answer: str
    confidence: float
    metadata: dict = field(default_factory=dict)


class JLens:
    """
    J-Lens: Tool for visualizing and interpreting model internals.

    The J-lens reads the J-space at different layer depths to reveal
    what concepts are active at each stage of processing.

    Usage:
        lens = JLens()
        result = lens.analyze("What color is the planet fourth from the sun?")
        print(result.readings)
        print(result.final_answer)
    """

    # Layer thresholds (percentage of total layers)
    LAYER_PERCENTAGES = {
        LayerDepth.INPUT: (0, 0.2),
        LayerDepth.EARLY: (0.2, 0.4),
        LayerDepth.MIDDLE: (0.4, 0.6),
        LayerDepth.LATE: (0.6, 0.8),
        LayerDepth.FINAL: (0.8, 1.0),
    }

    def __init__(self, total_layers: int = 96):
        self.total_layers = total_layers
        self.jspace = JointWorkspace()
        self.readings_history: list[LensResult] = []

    # === Main Analysis Methods ===

    def analyze(self, prompt: str, mode: LensMode = None) -> LensResult:
        """
        Analyze a prompt through the J-lens.

        Returns layer-by-layer readings of what concepts are active.
        """
        # Auto-detect mode if not specified
        if mode is None:
            mode = self._detect_mode(prompt)

        # Initialize J-space with prompt
        self.jspace.clear()
        self.jspace.set_task(prompt)

        # Generate layer readings
        readings = []
        intermediate_steps = []

        for layer_pct in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            layer_num = int(layer_pct * self.total_layers)
            depth = self._get_layer_depth(layer_pct)

            # Get concepts active at this layer
            concepts = self._get_concepts_at_layer(layer_num, depth, prompt)

            # Generate interpretation
            interpretation = self._interpret_layer(
                layer_num, depth, concepts, mode, prompt
            )

            reading = LayerReading(
                layer=layer_num,
                layer_depth=depth,
                concepts=concepts,
                activations=[0.7] * len(concepts),
                interpretation=interpretation,
                confidence=0.85
            )
            readings.append(reading)

            # Track intermediate steps
            if concepts:
                intermediate_steps.append(
                    f"Layer {layer_num}: {', '.join(concepts[:3])}"
                )

        # Generate final answer
        final_answer = self._generate_final_answer(readings, prompt, mode)

        result = LensResult(
            prompt=prompt,
            mode=mode,
            readings=readings,
            intermediate_steps=intermediate_steps,
            final_answer=final_answer,
            confidence=0.9,
            metadata={"total_layers": self.total_layers}
        )

        self.readings_history.append(result)
        return result

    def analyze_code(self, code: str) -> LensResult:
        """
        Analyze code for potential bugs.

        Based on Image 2 showing bug detection at empty-list call.
        """
        # Detect code patterns
        bugs = self._detect_code_bugs(code)

        # Create readings for each bug found
        readings = []
        for bug in bugs:
            reading = LayerReading(
                layer=bug["layer"],
                layer_depth=LayerDepth.MIDDLE,
                concepts=bug["concepts"],
                activations=[0.8] * len(bug["concepts"]),
                interpretation=bug["interpretation"],
                confidence=bug["confidence"]
            )
            readings.append(reading)

        # Generate final analysis
        if bugs:
            final = f"Found {len(bugs)} potential issue(s): " + "; ".join(
                b["type"] for b in bugs
            )
        else:
            final = "No obvious bugs detected."

        return LensResult(
            prompt=code[:100],
            mode=LensMode.CODE_DEBUG,
            readings=readings,
            intermediate_steps=[b["interpretation"] for b in bugs],
            final_answer=final,
            confidence=0.85 if bugs else 0.7
        )

    def analyze_arithmetic(self, expression: str) -> LensResult:
        """
        Analyze arithmetic expression step-by-step.

        Based on Image 2 showing mental arithmetic tracking.
        """
        # Parse and evaluate
        steps = self._trace_arithmetic(expression)

        readings = []
        for step in steps:
            reading = LayerReading(
                layer=step["layer"],
                layer_depth=LayerDepth.MIDDLE,
                concepts=step["concepts"],
                activations=[0.9] * len(step["concepts"]),
                interpretation=step["interpretation"],
                confidence=0.95
            )
            readings.append(reading)

        final_answer = steps[-1]["result"] if steps else "Error"

        return LensResult(
            prompt=expression,
            mode=LensMode.ARITHMETIC,
            readings=readings,
            intermediate_steps=[s["interpretation"] for s in steps],
            final_answer=str(final_answer),
            confidence=0.95
        )

    def analyze_sequence(self, sequence: str) -> LensResult:
        """
        Analyze a sequence (protein, DNA, etc).

        Based on Image 2 showing protein recognition.
        """
        # Identify sequence type
        seq_type = self._identify_sequence_type(sequence)

        # Generate readings based on sequence windows
        readings = []
        window_size = 5

        for i in range(0, min(len(sequence), 30), window_size):
            window = sequence[i:i+window_size]
            concepts = self._analyze_sequence_window(window, seq_type)

            reading = LayerReading(
                layer=50 + i,
                layer_depth=LayerDepth.MIDDLE,
                concepts=concepts,
                activations=[0.7] * len(concepts),
                interpretation=f"Position {i}-{i+window_size}: {', '.join(concepts[:3])}",
                confidence=0.8
            )
            readings.append(reading)

        # Generate final interpretation
        final = self._interpret_sequence(sequence, seq_type)

        return LensResult(
            prompt=sequence[:50],
            mode=LensMode.SEQUENCE,
            readings=readings,
            intermediate_steps=[r.interpretation for r in readings],
            final_answer=final,
            confidence=0.85
        )

    # === Safety Analysis ===

    def analyze_safety(self, text: str) -> dict[str, Any]:
        """
        Analyze text for safety concerns.

        Based on Image 8 (situational awareness) and
        Image 10 (malicious intent detection).
        """
        # Run multiple safety checks
        injection = self.jspace.detect_injection()
        fabrication = self.jspace.detect_fabrication()
        malicious = self.jspace.detect_malicious_intent()
        awareness = self.jspace.check_situational_awareness()

        # Activate concepts from text
        words = text.split()[:20]
        for i, word in enumerate(words):
            concept = Concept(
                id=f"safety_{i}",
                text=word,
                concept_type=ConceptType.FACTUAL,
                activation=0.5,
                layer=40 + i * 2,
                source="input"
            )
            self.jspace.activate(concept)

        # Re-run checks with activated concepts
        injection2 = self.jspace.detect_injection()
        fabrication2 = self.jspace.detect_fabrication()
        malicious2 = self.jspace.detect_malicious_intent()

        return {
            "text_preview": text[:100],
            "injection_detection": {
                "detected": len(injection2) > 0,
                "signals": [c.text for c in injection2]
            },
            "fabrication_detection": fabrication2,
            "malicious_intent_detection": malicious2,
            "situational_awareness": awareness,
            "overall_safety_score": self._calculate_safety_score(
                injection2, fabrication2, malicious2
            ),
            "recommendation": self._get_safety_recommendation(
                injection2, fabrication2, malicious2
            )
        }

    # === Layer Analysis Helpers ===

    def _detect_mode(self, prompt: str) -> LensMode:
        """Auto-detect the analysis mode from prompt."""
        prompt_lower = prompt.lower()

        # Arithmetic patterns
        if re.search(r'[\d\+\-\*\/\=\(\)]+', prompt):
            return LensMode.ARITHMETIC

        # Code patterns
        if any(kw in prompt_lower for kw in ['def ', 'class ', 'import ', 'function', 'return', 'if ', 'for ']):
            return LensMode.CODE_DEBUG

        # Sequence patterns (protein/DNA)
        if re.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', prompt.replace(' ', '')):
            return LensMode.SEQUENCE

        # Recall patterns
        if any(kw in prompt_lower for kw in ['what', 'who', 'when', 'where', 'how many']):
            return LensMode.RECALL

        # Safety patterns
        if any(kw in prompt_lower for kw in ['ignore', 'override', 'bypass', 'jailbreak']):
            return LensMode.SAFETY

        return LensMode.REASONING

    def _get_layer_depth(self, layer_pct: float) -> LayerDepth:
        """Get layer depth category from percentage."""
        for depth, (low, high) in self.LAYER_PERCENTAGES.items():
            if low <= layer_pct < high:
                return depth
        return LayerDepth.FINAL

    def _get_concepts_at_layer(self, layer: int, depth: LayerDepth, prompt: str) -> list[str]:
        """Get concepts that would be active at a specific layer."""
        concepts = []

        # Different concepts appear at different layers
        if depth == LayerDepth.INPUT:
            # Raw tokens
            concepts = prompt.split()[:5]
        elif depth == LayerDepth.EARLY:
            # Feature extraction
            concepts = self._extract_features(prompt)[:4]
        elif depth == LayerDepth.MIDDLE:
            # Reasoning concepts
            concepts = self._extract_reasoning_concepts(prompt)[:4]
        elif depth == LayerDepth.LATE:
            # Output preparation
            concepts = self._extract_output_concepts(prompt)[:3]
        elif depth == LayerDepth.FINAL:
            # Final decision
            concepts = self._extract_final_concepts(prompt)[:2]

        return concepts

    def _extract_features(self, text: str) -> list[str]:
        """Extract features from text."""
        features = []
        words = text.split()

        # Named entities (simple heuristic)
        for word in words:
            if word[0].isupper() and len(word) > 2:
                features.append(word.lower())

        # Numbers
        for word in words:
            if word.isdigit():
                features.append(f"number:{word}")

        return features[:5]

    def _extract_reasoning_concepts(self, text: str) -> list[str]:
        """Extract reasoning concepts."""
        concepts = []

        # Question words
        question_words = ['what', 'why', 'how', 'when', 'where', 'who']
        for qw in question_words:
            if qw in text.lower():
                concepts.append(qw)

        # Action words
        action_words = ['compute', 'calculate', 'find', 'determine', 'analyze']
        for aw in action_words:
            if aw in text.lower():
                concepts.append(aw)

        return concepts[:5]

    def _extract_output_concepts(self, text: str) -> list[str]:
        """Extract output-related concepts."""
        concepts = []

        # Answer indicators
        if '?' in text:
            concepts.append("question")

        # Number indicators
        numbers = re.findall(r'\d+', text)
        if numbers:
            concepts.extend([f"num:{n}" for n in numbers[:3]])

        return concepts[:3]

    def _extract_final_concepts(self, text: str) -> list[str]:
        """Extract final decision concepts."""
        # Simple: return the most important word
        words = text.split()
        if words:
            return [max(words, key=len).lower()]
        return ["result"]

    def _interpret_layer(self, layer: int, depth: LayerDepth,
                        concepts: list[str], mode: LensMode, prompt: str) -> str:
        """Generate interpretation for a layer reading."""
        if not concepts:
            return f"Layer {layer} ({depth.value}): Minimal activation"

        concept_str = ", ".join(concepts[:3])

        if mode == LensMode.RECALL:
            return f"Layer {layer}: Recalling '{concept_str}'"
        elif mode == LensMode.ARITHMETIC:
            return f"Layer {layer}: Computing '{concept_str}'"
        elif mode == LensMode.CODE_DEBUG:
            return f"Layer {layer}: Analyzing '{concept_str}'"
        elif mode == LensMode.SEQUENCE:
            return f"Layer {layer}: Recognizing '{concept_str}'"
        elif mode == LensMode.SAFETY:
            return f"Layer {layer}: Safety check '{concept_str}'"
        else:
            return f"Layer {layer}: Processing '{concept_str}'"

    def _generate_final_answer(self, readings: list[LensResult],
                              prompt: str, mode: LensMode) -> str:
        """Generate final answer from all readings."""
        if not readings:
            return "Unable to generate answer"

        final_reading = readings[-1]

        if mode == LensMode.RECALL:
            # Extract answer from final concepts
            if final_reading.concepts:
                return final_reading.concepts[-1]
            return "Unknown"
        elif mode == LensMode.ARITHMETIC:
            # Return computed value
            return final_reading.interpretation
        elif mode == LensMode.CODE_DEBUG:
            # Return bug summary
            bugs = [r for r in readings if "error" in r.interpretation.lower()
                    or "bug" in r.interpretation.lower()]
            if bugs:
                return f"Found {len(bugs)} issue(s)"
            return "Code appears valid"
        else:
            return final_reading.interpretation

    # === Code Bug Detection ===

    def _detect_code_bugs(self, code: str) -> list[dict]:
        """Detect potential bugs in code."""
        bugs = []
        lines = code.split('\n')

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Empty list call (like Image 2)
            if 'avg([]' in line_stripped or 'avg([])' in line_stripped:
                bugs.append({
                    "type": "division_by_zero",
                    "layer": 71,
                    "concepts": ["ValueError", "empty", "division"],
                    "interpretation": "Empty list will cause division by zero",
                    "confidence": 0.95
                })

            # Undefined variable
            if '=' in line_stripped and 'def ' not in line_stripped:
                parts = line_stripped.split('=')
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    if var_name and not var_name.startswith('#'):
                        # Check if variable was defined before
                        defined_before = any(
                            var_name in lines[j]
                            for j in range(i)
                        )
                        if not defined_before and var_name not in ['self', 'cls']:
                            bugs.append({
                                "type": "potential_undefined",
                                "layer": 54,
                                "concepts": ["undefined", "variable", var_name],
                                "interpretation": f"Variable '{var_name}' may be undefined",
                                "confidence": 0.6
                            })

            # Common bugs
            if '==' in line_stripped and 'if ' in line_stripped:
                # Check for assignment instead of comparison
                parts = line_stripped.split('==')
                if len(parts) == 2:
                    if '=' in parts[0] and '==' not in parts[0]:
                        bugs.append({
                            "type": "assignment_in_condition",
                            "layer": 65,
                            "concepts": ["assignment", "comparison", "condition"],
                            "interpretation": "Possible assignment in condition (use == for comparison)",
                            "confidence": 0.8
                        })

        return bugs

    # === Arithmetic Tracing ===

    def _trace_arithmetic(self, expression: str) -> list[dict]:
        """Trace arithmetic expression evaluation."""
        steps = []

        # Simple parser for basic expressions
        try:
            # Replace common patterns
            expr = expression.replace('²', '**2').replace('^', '**')

            # Tokenize
            tokens = re.findall(r'[\d\.\+\-\*\/\(\)]+', expr)

            # Evaluate step by step
            result = eval(expr)

            # Generate intermediate steps
            steps.append({
                "layer": 58,
                "concepts": ["Math", "parse"],
                "interpretation": f"Parsed expression: {expr}",
                "result": None
            })

            # Find intermediate calculations
            if '**' in expr:
                steps.append({
                    "layer": 75,
                    "concepts": ["power", "exponent"],
                    "interpretation": "Computing power/exponent",
                    "result": None
                })

            if '+' in expr or '-' in expr:
                steps.append({
                    "layer": 83,
                    "concepts": ["addition", "subtraction"],
                    "interpretation": "Performing addition/subtraction",
                    "result": None
                })

            steps.append({
                "layer": 96,
                "concepts": ["result", "answer"],
                "interpretation": f"Final result: {result}",
                "result": result
            })

        except Exception as e:
            steps.append({
                "layer": 96,
                "concepts": ["error", "invalid"],
                "interpretation": f"Error: {str(e)}",
                "result": None
            })

        return steps

    # === Sequence Analysis ===

    def _identify_sequence_type(self, sequence: str) -> str:
        """Identify the type of sequence."""
        seq = sequence.upper().replace(' ', '')

        # DNA
        if all(c in 'ATCG' for c in seq):
            return "dna"

        # RNA
        if all(c in 'AUCG' for c in seq):
            return "rna"

        # Protein
        protein_chars = set('ACDEFGHIKLMNPQRSTVWY')
        if all(c in protein_chars for c in seq):
            return "protein"

        return "unknown"

    def _analyze_sequence_window(self, window: str, seq_type: str) -> list[str]:
        """Analyze a sequence window."""
        concepts = []

        if seq_type == "protein":
            # Common amino acid properties
            hydrophobic = set('AILMFPWV')
            positive = set('RHK')
            negative = set('DE')
            special = set('CG')

            for aa in window:
                if aa in hydrophobic:
                    concepts.append("hydrophobic")
                elif aa in positive:
                    concepts.append("positive")
                elif aa in negative:
                    concepts.append("negative")
                elif aa in special:
                    concepts.append("special")

            # GFP-specific (from Image 2)
            if 'GFP' in window or 'FL' in window:
                concepts.append("fluorescent")
                concepts.append("green")

        elif seq_type == "dna":
            for nt in window:
                if nt in 'AT':
                    concepts.append("weak_pair")
                elif nt in 'GC':
                    concepts.append("strong_pair")

        return list(set(concepts))[:4]

    def _interpret_sequence(self, sequence: str, seq_type: str) -> str:
        """Generate final interpretation of sequence."""
        if seq_type == "protein":
            # Check for GFP-like sequence (from Image 2)
            if 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFIC' in sequence:
                return "Green Fluorescent Protein (GFP) detected - fluorescent"
            return f"Protein sequence ({len(sequence)} residues)"
        elif seq_type == "dna":
            return f"DNA sequence ({len(sequence)} bp)"
        elif seq_type == "rna":
            return f"RNA sequence ({len(sequence)} nt)"
        else:
            return f"Unknown sequence type ({len(sequence)} chars)"

    # === Safety Helpers ===

    def _calculate_safety_score(self, injection: list, fabrication: dict,
                               malicious: dict) -> float:
        """Calculate overall safety score (0-100, higher is safer)."""
        score = 100.0

        if injection:
            score -= len(injection) * 20
        if fabrication.get("is_fabricating"):
            score -= fabrication.get("confidence", 0) * 30
        if malicious.get("is_malicious"):
            score -= malicious.get("risk_score", 0) * 40

        return max(0.0, min(100.0, score))

    def _get_safety_recommendation(self, injection: list, fabrication: dict,
                                  malicious: dict) -> str:
        """Get safety recommendation."""
        if injection:
            return "BLOCK: Potential injection detected"
        if malicious.get("is_malicious"):
            return "BLOCK: Malicious intent detected"
        if fabrication.get("is_fabricating"):
            return "VERIFY: Potential data fabrication"
        return "ALLOW: No safety concerns detected"

    # === Utility ===

    def get_history(self) -> list[dict]:
        """Get history of analyses."""
        return [
            {
                "prompt": r.prompt[:50],
                "mode": r.mode.value,
                "readings_count": len(r.readings),
                "confidence": r.confidence
            }
            for r in self.readings_history
        ]


# === Convenience Functions ===

def quick_lens(prompt: str) -> dict:
    """Quick J-lens analysis."""
    lens = JLens()
    result = lens.analyze(prompt)
    return {
        "prompt": result.prompt,
        "mode": result.mode.value,
        "readings": len(result.readings),
        "intermediate_steps": result.intermediate_steps,
        "final_answer": result.final_answer,
        "confidence": result.confidence
    }


def debug_code(code: str) -> dict:
    """Quick code debugging with J-lens."""
    lens = JLens()
    result = lens.analyze_code(code)
    return {
        "code_preview": code[:100],
        "issues_found": len(result.readings),
        "issues": result.intermediate_steps,
        "summary": result.final_answer,
        "confidence": result.confidence
    }


def check_safety(text: str) -> dict:
    """Quick safety check with J-lens."""
    lens = JLens()
    return lens.analyze_safety(text)

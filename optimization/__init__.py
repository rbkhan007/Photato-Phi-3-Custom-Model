"""Optimization module for the custom Claude-like model.

Provides inference optimization, attention optimization, probability sampling,
vector operations, batch processing, graph optimization, memory-efficient loops,
and parallel processing utilities.
"""

from .attention import AttentionConfig, AttentionOptimizer, GroupedQueryAttention
from .batch_processor import BatchProcessor, DynamicBatcher, SequencePacker
from .graph_optimizer import ComputationGraph, GraphOptimizer, ModelGraphBuilder
from .inference_engine import OptimizedInference, SpeculativeDecoder, ContinuousBatching
from .memory_loops import MemoryEfficientLoop, MemoryMappedLoop
from .parallel import ParallelProcessor, Pipeline, AsyncProcessor
from .probability import ProbabilitySampler, TypicalSampler, MirostatSampler
from .vector_ops import VectorOptimizer, ANNIndex, VectorQuantizer

"""Full test coverage for the optimization package (pure-Python, no ML deps)."""
import math
import pytest

from optimization.vector_ops import VectorOptimizer, ANNIndex, VectorQuantizer
from optimization.probability import (
    ProbabilitySampler,
    SamplingConfig,
    TypicalSampler,
    MirostatSampler,
)
from optimization.attention import (
    AttentionConfig,
    AttentionOptimizer,
    GroupedQueryAttention,
)
from optimization.batch_processor import (
    BatchProcessor,
    BatchItem,
    BatchResult,
    DynamicBatcher,
    SequencePacker,
    MemoryEfficientBatcher,
)
from optimization.graph_optimizer import (
    GraphOptimizer,
    ComputationGraph,
    ModelGraphBuilder,
)
from optimization.inference_engine import (
    OptimizedInference,
    KVCache,
    FastTokenizer,
    SpeculativeDecoder,
    ContinuousBatching,
)
from optimization.memory_loops import MemoryEfficientLoop, BufferPool, RecursiveLoop
from optimization.parallel import (
    ParallelProcessor,
    WorkStealingQueue,
    WorkItem,
    Pipeline,
    AsyncProcessor,
)


# ----------------------------- vector_ops -----------------------------
def test_vector_dot_product():
    v = VectorOptimizer(dimension=3)
    assert v.dot_product([1, 2, 3], [4, 5, 6]) == 32


def test_vector_cosine_identical():
    v = VectorOptimizer(dimension=2)
    assert v.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_vector_cosine_orthogonal():
    v = VectorOptimizer(dimension=2)
    assert v.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_vector_l2_distance():
    v = VectorOptimizer(dimension=2)
    assert v.l2_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)


def test_vector_manhattan():
    v = VectorOptimizer(dimension=2)
    assert v.manhattan_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(7.0)


def test_vector_normalize():
    v = VectorOptimizer(dimension=2)
    n = v.normalize([3.0, 4.0])
    assert n[0] == pytest.approx(0.6)
    assert n[1] == pytest.approx(0.8)


def test_vector_norm():
    v = VectorOptimizer(dimension=2)
    assert v.vector_norm([3.0, 4.0]) == pytest.approx(5.0)


def test_vector_add_and_find_similar():
    v = VectorOptimizer(dimension=2)
    v.add_vectors([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
    results = v.find_similar([1.0, 0.0], top_k=1)
    assert results and results[0][0] == "a"


def test_vector_batch_cosine():
    v = VectorOptimizer(dimension=2)
    res = v.batch_cosine_similarity([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    assert res[0] == pytest.approx(1.0)
    assert res[1] == pytest.approx(0.0)


def test_ann_index_search():
    idx = ANNIndex(dimension=2, num_lists=1)
    idx.add("a", [1.0, 0.0])
    idx.add("b", [0.0, 1.0])
    results = idx.search([1.0, 0.0], top_k=1)
    assert results and results[0][0] == "a"


def test_vector_quantizer_roundtrip():
    q = VectorQuantizer(codebook_size=4)
    data = [[0.1, 0.2], [0.9, 0.8], [0.15, 0.25], [0.85, 0.75]]
    q.train(data)
    codes = [q.quantize(v) for v in data]
    deq = [q.dequantize(c) for c in codes]
    assert len(deq) == len(data)
    assert len(deq[0]) == 2


# ----------------------------- probability -----------------------------
def test_probability_sample_returns_known_token():
    sampler = ProbabilitySampler(SamplingConfig(seed=42, temperature=0.1))
    token = sampler.sample({"cat": 2.0, "dog": 0.1, "bird": 0.1})
    assert token in {"cat", "dog", "bird"}


def test_probability_logits_to_probs():
    sampler = ProbabilitySampler()
    probs = sampler.logits_to_probs({"a": 0.0, "b": 0.0})
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_probability_top_k():
    sampler = ProbabilitySampler()
    probs = sampler.apply_top_k({"a": 1.0, "b": 0.5, "c": 0.2}, k=2)
    assert set(probs.keys()) == {"a", "b"}


def test_probability_top_p():
    sampler = ProbabilitySampler()
    probs = sampler.apply_top_p({"a": 0.6, "b": 0.3, "c": 0.1}, p=0.7)
    assert set(probs.keys()) == {"a", "b"}


def test_probability_temperature():
    sampler = ProbabilitySampler()
    probs = sampler.apply_temperature({"a": 1.0, "b": 1.0}, temperature=0.5)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_probability_repeat_penalty():
    sampler = ProbabilitySampler()
    probs = sampler.apply_repeat_penalty({"a": 1.0, "b": 1.0}, ["a"], penalty=1.5)
    assert probs["a"] < probs["b"]


def test_probability_get_top_tokens():
    sampler = ProbabilitySampler()
    top = sampler.get_top_tokens({"a": 1.0, "b": 0.5, "c": 0.2}, n=2)
    assert [t for t, _ in top] == ["a", "b"]


def test_typical_sampler():
    s = TypicalSampler()
    token = s.sample({"a": 1.0, "b": 0.5})
    assert token in {"a", "b"}


def test_mirostat_sampler():
    s = MirostatSampler(tau=5.0, eta=0.1)
    token = s.sample({"a": 1.0, "b": 0.5, "c": 0.2})
    assert token in {"a", "b", "c"}


# ----------------------------- attention -----------------------------
def test_attention_multi_head_shape():
    cfg = AttentionConfig(num_heads=2, head_dim=4)
    opt = AttentionOptimizer(cfg)
    seq = [[1.0] * 8, [2.0] * 8]
    out = opt.multi_head_attention(seq, seq, seq)
    assert isinstance(out, list)
    assert len(out) == 2
    assert len(out[0]) == 8


def test_attention_sparse_shape():
    cfg = AttentionConfig(num_heads=1, head_dim=4)
    opt = AttentionOptimizer(cfg)
    seq = [[1.0] * 4, [2.0] * 4, [3.0] * 4]
    out = opt.sparse_attention(seq, seq, seq)
    assert isinstance(out, list)
    assert len(out) == 3


def test_gqa_expand_kv():
    gqa = GroupedQueryAttention(num_heads=2, num_kv_heads=1, head_dim=2)
    expanded = gqa._expand_kv([[1.0, 2.0]])
    assert isinstance(expanded, list) and len(expanded) >= 1


def test_gqa_forward_shape():
    gqa = GroupedQueryAttention(num_heads=2, num_kv_heads=1, head_dim=2)
    q = [[1.0, 2.0, 3.0, 4.0]]
    out = gqa.forward(query=q, key=q, value=q)
    assert isinstance(out, list)
    assert len(out) == 1
    assert len(out[0]) == gqa.head_dim


# ----------------------------- batch_processor -----------------------------
def test_batch_processor_add_and_process():
    bp = BatchProcessor()
    bp.add_item("1", "hello world", length=2)
    size = bp.get_optimal_batch_size([])
    assert isinstance(size, int) and size >= 1
    items = [BatchItem(id="1", data="hello", length=2)]
    result = bp.process_batch(lambda batch: [item.data.upper() for item in batch], items)
    assert isinstance(result, BatchResult)
    assert len(result.results) == 1
    flat = [x for batch in result.results for x in (batch if isinstance(batch, list) else [batch])]
    assert "HELLO" in flat


def test_batch_packer_pack():
    packed = SequencePacker.pack([["hello"], ["world"], ["foo"]], max_length=10)
    assert isinstance(packed, list)


def test_dynamic_batcher():
    db = DynamicBatcher(target_batch_time=0.05)
    new_size = db.update_batch_size(0.05, 8)
    assert isinstance(new_size, int) and new_size >= 1


def test_memory_efficient_batcher():
    mb = MemoryEfficientBatcher(max_memory_mb=1000)
    steps = mb.calculate_accumulation_steps(32, 8)
    assert isinstance(steps, int) and steps >= 1
    assert mb.get_effective_batch_size(8) >= 8


# ----------------------------- graph_optimizer -----------------------------
def test_graph_optimizer_optimize_empty():
    go = GraphOptimizer()
    result = go.optimize(ComputationGraph())
    assert isinstance(result, ComputationGraph)


def test_graph_optimizer_build_and_optimize():
    go = GraphOptimizer()
    g = ModelGraphBuilder.build_transformer_layer(layer_id=0, hidden_dim=64, num_heads=2)
    result = go.optimize(g)
    assert isinstance(result, ComputationGraph)
    report = go.get_optimization_report()
    assert isinstance(report, dict)


def test_graph_fuse_constants():
    go = GraphOptimizer()
    g = ComputationGraph()
    out = go.fold_constants(g)
    assert isinstance(out, ComputationGraph)


def test_graph_eliminate_dead_code():
    go = GraphOptimizer()
    g = ComputationGraph()
    out = go.eliminate_dead_code(g)
    assert isinstance(out, ComputationGraph)


# ----------------------------- inference_engine -----------------------------
def test_optimized_inference_generate():
    eng = OptimizedInference(model_path="")
    text = eng.generate("hello world", max_tokens=3)
    assert isinstance(text, str)


def test_kv_cache():
    cache = KVCache()
    cache.append([0.1, 0.2], [0.3, 0.4])
    assert len(cache.keys) == 1
    assert cache.keys[-1] == [0.1, 0.2]
    assert cache.values[-1] == [0.3, 0.4]
    keys, values = cache.get()
    assert keys[-1] == [0.1, 0.2]
    cache.clear()
    assert len(cache.keys) == 0


def test_fast_tokenizer():
    assert FastTokenizer.count_tokens("hello world foo") >= 1
    chunks = FastTokenizer.split_into_chunks("a b c d e", 2)
    assert isinstance(chunks, list)


def test_speculative_decoder_construct():
    sd = SpeculativeDecoder(draft_model_path="", target_model_path="")
    assert sd is not None


def test_continuous_batching():
    cb = ContinuousBatching(max_batch_size=4)
    cb.add_request("r1", "hello")
    cb.add_request("r2", "world")
    results = cb.process_batch()
    assert isinstance(results, list)


# ----------------------------- memory_loops -----------------------------
def test_memory_loop_chunked_iter():
    loop = MemoryEfficientLoop()
    chunks = list(loop.chunked_iter(list(range(10)), chunk_size=3))
    flat = [x for c in chunks for x in c]
    assert flat == list(range(10))


def test_memory_loop_map_reduce():
    loop = MemoryEfficientLoop()
    total = loop.map_reduce(list(range(5)), lambda x: x * 2, lambda results: sum(results))
    assert total == sum(x * 2 for x in range(5))


def test_memory_loop_sliding_window():
    loop = MemoryEfficientLoop()
    wins = list(loop.sliding_window([1, 2, 3, 4, 5], window_size=2))
    assert wins[0] == [1, 2]


def test_memory_loop_lazy_filter_transform():
    loop = MemoryEfficientLoop()
    out = loop.lazy_transform(
        loop.lazy_filter(range(10), lambda x: x % 2 == 0), lambda x: x * 10
    )
    assert list(out) == [0, 20, 40, 60, 80]


def test_memory_loop_take_consume():
    loop = MemoryEfficientLoop()
    gen = iter(range(100))
    assert list(loop.take(gen, 3)) == [0, 1, 2]
    loop.consume(range(5))


def test_buffer_pool():
    pool = BufferPool(buffer_size=10)
    buf = pool.acquire(8)
    assert buf is not None
    pool.release(buf)


def test_recursive_loop_flatten():
    rl = RecursiveLoop()
    assert list(rl.flatten([1, [2, [3, 4]], 5])) == [1, 2, 3, 4, 5]


# ----------------------------- parallel -----------------------------
def test_parallel_map():
    pp = ParallelProcessor()
    out = pp.parallel_map(lambda x: x * 2, [1, 2, 3])
    assert set(out) == {2, 4, 6}


def test_work_stealing_queue():
    q = WorkStealingQueue()
    q.add(0, WorkItem(id=1, data="a"))
    stolen = q.steal(99)
    assert stolen is not None and stolen.data == "a"


def test_pipeline():
    pipe = Pipeline([lambda x: x + 1, lambda x: x * 2])
    out = pipe.process([1, 2, 3])
    assert set(out) == {4, 6, 8}


def test_async_processor():
    ap = AsyncProcessor()
    thread = ap.process_async("t1", lambda x: [i + 1 for i in x], [1, 2, 3])
    thread.join()
    result = ap.get_result("t1")
    assert set(result) == {2, 3, 4}

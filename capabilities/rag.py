#!/usr/bin/env python3
"""
RAG (Retrieval Augmented Generation) for Local LLMs.

Features:
- Document ingestion and chunking
- Vector embeddings (local GGUF or TF-IDF fallback)
- Semantic search (pgvector or in-memory)
- Context-aware generation
- Batch operations with progress reporting
- Embedding caching
- PostgreSQL pgvector with HNSW/IVFFlat indexes

Usage:
    from capabilities.rag import RAGEngine

    rag = RAGEngine()
    rag.ingest_document("path/to/doc.txt")
    response = rag.query("What is the main topic?")
"""

import hashlib
import json
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"
DEFAULT_EMBED_MODEL = str(_NOTEBOOKS / "Qwen3-Embedding-0.6B-Q8_0.gguf")
DEFAULT_GEN_MODEL = str(_NOTEBOOKS / "Phi-4-mini-instruct-Q4_K_M.gguf")


@dataclass
class Document:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    chunks: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    document_id: str
    content: str
    index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[Chunk]
    scores: list[float]
    context: str
    metadata: dict = field(default_factory=dict)


class TextChunker:
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
    ) -> list[str]:
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = []
        current_chunk = ""
        for separator in separators:
            if separator == "":
                for i in range(0, len(text), chunk_size - chunk_overlap):
                    chunk = text[i:i + chunk_size]
                    if chunk.strip():
                        chunks.append(chunk)
                break
            parts = text.split(separator)
            for part in parts:
                if len(current_chunk) + len(part) + len(separator) <= chunk_size:
                    current_chunk += part + separator
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = part + separator
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            if len(chunks) > 1:
                break
        return [c for c in chunks if c.strip()]

    @staticmethod
    def chunk_by_sentence(text: str, max_sentences: int = 5) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk = " ".join(sentences[i:i + max_sentences])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    @staticmethod
    def chunk_by_paragraph(text: str) -> list[str]:
        return [p.strip() for p in text.split("\n\n") if p.strip()]


class EmbeddingCache:
    def __init__(self, capacity: int = 1024):
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, text: str) -> Optional[list[float]]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, text: str, vector: list[float]):
        key = hashlib.md5(text.encode()).hexdigest()
        self._cache[key] = vector
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class SimpleEmbeddings:
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def embed(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        vector = [0.0] * self.dimension
        for token, count in tf.items():
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension
            tf_score = count / len(tokens) if tokens else 0
            vector[idx] += tf_score * 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class GgufEmbedder:
    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = 0, cpu_percent: float = 55.0, verbose: bool = False):
        from llama_cpp import Llama
        from optimization.cpu_throttle import limit_cpu
        self.model_path = str(model_path)
        self.cpu_percent = cpu_percent
        threads = limit_cpu(cpu_percent)
        self._threads = threads
        self._model = Llama(
            model_path=self.model_path,
            embedding=True,
            n_ctx=n_ctx,
            n_threads=threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        raw_dim = getattr(self._model, "n_embd", None) or getattr(self._model, "embedding_dim", None)
        if callable(raw_dim):
            try:
                raw_dim = raw_dim()
            except Exception:
                raw_dim = None
        self.dimension = int(raw_dim) if raw_dim else len(self.embed("dimension probe"))
        self._cache = EmbeddingCache(capacity=2048)

    def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vec = self._model.embed(text, normalize=True, truncate=True)
        result = [float(x) for x in vec]
        self._cache.put(text, result)
        return result

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_batch(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        results = []
        for i, t in enumerate(texts):
            results.append(self.embed(t))
            if show_progress and (i + 1) % 10 == 0:
                print(f"\r[rag] embedding {i + 1}/{len(texts)}", end="", flush=True)
        if show_progress and len(texts) > 10:
            print()
        return results

    @property
    def cache_stats(self) -> dict:
        return {"size": len(self._cache._cache), "hit_rate": self._cache.hit_rate}


class VectorStore:
    def __init__(self):
        self.vectors: dict[str, list[float]] = {}
        self.metadata: dict[str, dict] = {}
        self._index_built = False
        self._partitions: dict[int, list[str]] = {}
        self._num_partitions: int = 8

    def add(self, id: str, vector: list[float], meta: Optional[dict] = None):
        self.vectors[id] = vector
        if meta:
            self.metadata[id] = meta
        self._index_built = False

    def add_batch(self, items: list[tuple[str, list[float], Optional[dict]]]):
        for id, vec, meta in items:
            self.vectors[id] = vec
            if meta:
                self.metadata[id] = meta
        self._index_built = False

    def _build_partitions(self):
        if self._index_built or len(self.vectors) < 32:
            return
        ids = list(self.vectors.keys())
        for i, vid in enumerate(ids):
            part = i % self._num_partitions
            self._partitions.setdefault(part, []).append(vid)
        self._index_built = True

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        if len(vec1) != len(vec2):
            return 0.0
        dot = 0.0
        n1 = 0.0
        n2 = 0.0
        for a, b in zip(vec1, vec2):
            dot += a * b
            n1 += a * a
            n2 += b * b
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (math.sqrt(n1) * math.sqrt(n2))

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        scores = []
        for id, vector in self.vectors.items():
            score = self._cosine_similarity(query_vector, vector)
            if score >= threshold:
                scores.append((id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def remove(self, id: str):
        self.vectors.pop(id, None)
        self.metadata.pop(id, None)

    def clear(self):
        self.vectors.clear()
        self.metadata.clear()
        self._partitions.clear()
        self._index_built = False

    @property
    def count(self) -> int:
        return len(self.vectors)

    def save(self, path: str):
        data = {"vectors": self.vectors, "metadata": self.metadata}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
            self.vectors = data.get("vectors", {})
            self.metadata = data.get("metadata", {})


class PGVectorStore:
    def __init__(
        self,
        dsn: Optional[str] = None,
        table: str = "rag_vectors",
        dimension: int = 128,
    ):
        self.dsn = dsn or os.environ.get("PGVECTOR_DSN") or os.environ.get("DATABASE_URL")
        self.table = self._safe_name(table)
        self.dimension = dimension
        self.metadata: dict[str, dict] = {}
        self._conn = None
        self._driver = None
        self._pool = None

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_]', '', name) or "rag_vectors"

    @staticmethod
    def available() -> bool:
        try:
            import psycopg
            return True
        except Exception:
            try:
                import psycopg2
                return True
            except Exception:
                return False

    def _connect(self):
        if self._conn is not None and not self._closed():
            return self._conn
        if self.dsn is None:
            raise RuntimeError("No PostgreSQL DSN configured. Set PGVECTOR_DSN or DATABASE_URL env var.")
        try:
            import psycopg
            self._driver = "psycopg"
            self._conn = psycopg.connect(self.dsn)
        except Exception:
            import psycopg2
            self._driver = "psycopg2"
            self._conn = psycopg2.connect(self.dsn)
        self._conn.autocommit = True
        return self._conn

    def _closed(self) -> bool:
        closed = getattr(self._conn, "closed", False)
        if isinstance(closed, bool):
            return closed
        return closed not in (False, 0)

    def _cursor(self):
        return self._connect().cursor()

    def ensure_schema(self):
        cur = self._cursor()
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id text PRIMARY KEY,
                    embedding vector({self.dimension}),
                    content text,
                    document_id text,
                    chunk_index integer,
                    metadata jsonb
                )
                """,
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_content ON {self.table} USING gin(to_tsvector('english', coalesce(content, '')))"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_doc_id ON {self.table}(document_id)"
            )
        finally:
            cur.close()

    def ensure_index(self, index_type: str = "hnsw"):
        cur = self._cursor()
        try:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            count = (cur.fetchone() or [0])[0]
            if count < 100:
                return
            idx_name = f"idx_{self.table}_embedding_{index_type}"
            cur.execute(f"SELECT 1 FROM pg_indexes WHERE indexname = '{idx_name}'")
            if cur.fetchone():
                return
            if index_type == "hnsw":
                cur.execute(f"""
                    CREATE INDEX {idx_name} ON {self.table}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 200)
                """)
            else:
                lists = max(1, count // 1000)
                cur.execute(f"""
                    CREATE INDEX {idx_name} ON {self.table}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {lists})
                """)
            print(f"[pgvector] Created {index_type} index on {self.table} ({count} rows)")
        except Exception as e:
            print(f"[pgvector] Index creation skipped: {e}")
        finally:
            cur.close()

    @staticmethod
    def _fmt(vec: list[float]) -> str:
        return "[" + ",".join(f"{float(x):.8g}" for x in vec) + "]"

    def add(self, id: str, vector: list[float], meta: Optional[dict] = None):
        meta = dict(meta or {})
        cur = self._cursor()
        try:
            cur.execute(
                f"""
                INSERT INTO {self.table} (id, embedding, content, document_id, chunk_index, metadata)
                VALUES (%s, %s::vector, %s, %s, %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    content = EXCLUDED.content,
                    document_id = EXCLUDED.document_id,
                    chunk_index = EXCLUDED.chunk_index,
                    metadata = EXCLUDED.metadata
                """,
                (
                    id,
                    self._fmt(vector),
                    meta.get("content", ""),
                    meta.get("document_id"),
                    meta.get("chunk_index"),
                    json.dumps(meta),
                ),
            )
        finally:
            cur.close()
        self.metadata[id] = meta

    def add_batch(self, items: list[tuple[str, list[float], Optional[dict]]]):
        cur = self._cursor()
        try:
            rows = []
            for id, vec, meta in items:
                meta = dict(meta or {})
                rows.append((
                    id,
                    self._fmt(vec),
                    meta.get("content", ""),
                    meta.get("document_id"),
                    meta.get("chunk_index"),
                    json.dumps(meta),
                ))
                self.metadata[id] = meta
            if self._driver == "psycopg2":
                from psycopg2.extras import execute_values
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.table} (id, embedding, content, document_id, chunk_index, metadata)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        content = EXCLUDED.content,
                        document_id = EXCLUDED.document_id,
                        chunk_index = EXCLUDED.chunk_index,
                        metadata = EXCLUDED.metadata
                    """,
                    rows,
                    template="(%s, %s::vector, %s, %s, %s, %s::jsonb)",
                )
            else:
                with cur.copy(
                    f"COPY {self.table} (id, embedding, content, document_id, chunk_index, metadata) FROM STDIN"
                ) as copy:
                    for row in rows:
                        copy.write_row(row)
        finally:
            cur.close()

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        cur = self._cursor()
        try:
            q = self._fmt(query_vector)
            if threshold > 0.0:
                cur.execute(
                    f"""
                    SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS score
                    FROM {self.table}
                    WHERE 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q, q, threshold, q, top_k),
                )
            else:
                cur.execute(
                    f"""
                    SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS score
                    FROM {self.table}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q, q, top_k),
                )
            rows = cur.fetchall()
        finally:
            cur.close()

        results = []
        for row in rows:
            rid, content, meta, score = row[0], row[1], row[2], row[3]
            if score is None:
                continue
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            cached = dict(meta) if isinstance(meta, dict) else {}
            if content is not None:
                cached["content"] = content
            self.metadata[rid] = cached
            results.append((rid, float(score)))
        return results[:top_k]

    def search_with_details(
        self,
        query_vector: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        results = self.search(query_vector, top_k, threshold)
        detailed = []
        for rid, score in results:
            meta = self.metadata.get(rid, {})
            detailed.append({
                "id": rid,
                "score": score,
                "content": meta.get("content", ""),
                "document_id": meta.get("document_id"),
                "chunk_index": meta.get("chunk_index"),
            })
        return detailed

    @property
    def vectors(self):
        class _Count:
            def __init__(self, store):
                self._store = store
            def __len__(self):
                cur = self._store._cursor()
                try:
                    cur.execute(f"SELECT count(*) FROM {self._store.table}")
                    return cur.fetchone()[0]
                finally:
                    cur.close()
        return _Count(self)

    def count(self) -> int:
        return len(self.vectors)

    def delete(self, id: str):
        cur = self._cursor()
        try:
            cur.execute(f"DELETE FROM {self.table} WHERE id = %s", (id,))
        finally:
            cur.close()
        self.metadata.pop(id, None)

    def clear(self):
        cur = self._cursor()
        try:
            cur.execute(f"TRUNCATE {self.table}")
        finally:
            cur.close()
        self.metadata.clear()

    def save(self, path: str):
        cur = self._cursor()
        try:
            cur.execute(
                f"SELECT id, embedding, content, document_id, chunk_index, metadata FROM {self.table}"
            )
            rows = cur.fetchall()
        finally:
            cur.close()
        data = []
        for rid, emb, content, doc_id, idx, meta in rows:
            emb_val = emb
            if isinstance(emb_val, str):
                try:
                    emb_val = json.loads(emb_val)
                except Exception:
                    emb_val = None
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            data.append({
                "id": rid,
                "embedding": list(emb_val) if emb_val is not None else None,
                "content": content,
                "document_id": doc_id,
                "chunk_index": idx,
                "metadata": meta if isinstance(meta, dict) else {},
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.add_batch([
            (
                row["id"],
                row["embedding"],
                dict(row.get("metadata") or {}, **{
                    "content": row.get("content", ""),
                    "document_id": row.get("document_id"),
                    "chunk_index": row.get("chunk_index", 0),
                }),
            )
            for row in data if row.get("embedding") is not None
        ])


class RAGEngine:
    def __init__(
        self,
        model_path: str = DEFAULT_GEN_MODEL,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
        embedding_dim: int = 1024,
        embedding_model: str = DEFAULT_EMBED_MODEL,
        embedding_n_gpu_layers: int = 0,
        cpu_percent: float = 55.0,
        vector_store: Optional[object] = None,
        vector_store_type: str = "auto",
        pg_dsn: Optional[str] = None,
    ):
        self.model_path = model_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.cpu_percent = cpu_percent
        self._generator = None

        self.chunker = TextChunker()
        self.embeddings = self._resolve_embedder(embedding_model, embedding_dim, embedding_n_gpu_layers, cpu_percent)
        try:
            self.embedding_dim = len(self.embeddings.embed("__dim_probe__"))
        except Exception:
            self.embedding_dim = embedding_dim
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = self._resolve_store(vector_store_type, pg_dsn, self.embedding_dim)
            if vector_store_type != "memory" and hasattr(self.vector_store, 'ensure_index'):
                try:
                    self.vector_store.ensure_index()
                except Exception:
                    pass
        self.documents: dict[str, Document] = {}
        self._ingestion_times: list[float] = []

    @staticmethod
    def _resolve_embedder(model_path: Optional[str], fallback_dim: int, n_gpu_layers: int, cpu_percent: float = 55.0):
        if model_path and Path(model_path).exists():
            try:
                emb = GgufEmbedder(model_path, n_gpu_layers=n_gpu_layers, cpu_percent=cpu_percent)
                print(f"[rag] Using GGUF embedder: {model_path} (dim={emb.dimension})")
                return emb
            except Exception as e:
                print(f"[rag] GGUF embedder unavailable ({e}); using local TF-IDF embeddings")
        return SimpleEmbeddings(dimension=fallback_dim)

    @staticmethod
    def _resolve_store(kind: str, dsn: Optional[str], dim: int):
        if kind in ("auto", "pgvector") and PGVectorStore.available():
            try:
                store = PGVectorStore(dsn=dsn, dimension=dim)
                store.ensure_schema()
                print("[rag] Using PostgreSQL + pgvector vector store")
                return store
            except Exception as e:
                if kind == "pgvector":
                    raise
                print(f"[rag] pgvector unavailable ({e}); falling back to in-memory store")
        return VectorStore()

    def _ingest_chunks(self, doc_id: str, chunks: list[str], metadata: dict):
        batch = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self.embeddings.embed(chunk_text)
            batch.append((
                chunk_id,
                embedding,
                {
                    "document_id": doc_id,
                    "chunk_index": i,
                    "content": chunk_text,
                    **metadata,
                },
            ))
        if hasattr(self.vector_store, 'add_batch'):
            self.vector_store.add_batch(batch)
        else:
            for item in batch:
                self.vector_store.add(*item)

    def ingest_document(self, file_path: str, metadata: Optional[dict] = None) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        content = path.read_text(encoding="utf-8")
        doc_id = hashlib.md5(content.encode()).hexdigest()
        doc = Document(
            id=doc_id,
            content=content,
            metadata={"source": str(path.absolute()), "filename": path.name, "size": len(content), **(metadata or {})},
        )
        doc.chunks = self.chunker.chunk_text(content, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        t0 = time.time()
        self._ingest_chunks(doc_id, doc.chunks, doc.metadata)
        elapsed = time.time() - t0
        self._ingestion_times.append(elapsed)
        self.documents[doc_id] = doc
        print(f"[rag] Ingested {path.name} ({len(doc.chunks)} chunks, {elapsed:.1f}s)")
        return doc

    def ingest_text(self, text: str, metadata: Optional[dict] = None) -> Document:
        doc_id = hashlib.md5(text.encode()).hexdigest()
        doc = Document(id=doc_id, content=text, metadata=metadata or {})
        doc.chunks = self.chunker.chunk_text(text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        t0 = time.time()
        self._ingest_chunks(doc_id, doc.chunks, doc.metadata)
        self._ingestion_times.append(time.time() - t0)
        self.documents[doc_id] = doc
        print(f"[rag] Ingested text ({len(doc.chunks)} chunks)")
        return doc

    def ingest_directory(self, dir_path: str, extensions: Optional[list[str]] = None) -> list[Document]:
        if extensions is None:
            extensions = [".txt", ".md", ".py", ".json"]
        documents = []
        path = Path(dir_path)
        files = [f for f in path.rglob("*") if f.suffix in extensions]
        for i, file_path in enumerate(files):
            try:
                doc = self.ingest_document(str(file_path))
                documents.append(doc)
                print(f"  [{i + 1}/{len(files)}] {file_path.name}")
            except Exception as e:
                print(f"  [x] {file_path.name}: {e}")
        return documents

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        top_k = top_k or self.top_k
        query_embedding = self.embeddings.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        chunks = []
        scores = []
        for chunk_id, score in results:
            meta = self.vector_store.metadata.get(chunk_id, {})
            chunks.append(Chunk(
                id=chunk_id,
                document_id=meta.get("document_id", ""),
                content=meta.get("content", ""),
                index=meta.get("chunk_index", 0),
                metadata=meta,
            ))
            scores.append(score)
        context = "\n\n---\n\n".join([c.content for c in chunks])
        return RetrievalResult(
            query=query, chunks=chunks, scores=scores, context=context,
            metadata={"num_chunks": len(chunks)},
        )

    def query(self, question: str, context_template: Optional[str] = None) -> str:
        retrieval = self.retrieve(question)
        if context_template is None:
            context_template = """Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer based on the context above. If the context doesn't contain enough information, say so."""
        prompt = context_template.format(context=retrieval.context, question=question)
        return self._generate_response(prompt, retrieval)

    def _generate_response(self, prompt: str, retrieval: RetrievalResult) -> str:
        if self._generator is None and self.model_path:
            try:
                from inference.llama_engine import FastLlamaEngine
                self._generator = FastLlamaEngine(self.model_path, cpu_percent=self.cpu_percent)
            except Exception as e:
                print(f"[rag] generation model unavailable ({e}); returning context-only stub")
                self._generator = False
        if self._generator:
            try:
                result = self._generator.generate(
                    prompt, max_tokens=512, temperature=0.1, repeat_penalty=1.15,
                )
                text = result.get("text", "") if isinstance(result, dict) else str(result)
                return text.strip()
            except Exception as e:
                print(f"[rag] generation failed ({e}); returning context-only stub")
                self._generator = False
        return f"Based on the retrieved context ({len(retrieval.chunks)} chunks), here is the answer..."

    def get_stats(self) -> dict:
        stats = {
            "documents": len(self.documents),
            "total_chunks": sum(len(d.chunks) for d in self.documents.values()),
            "vector_store_size": len(self.vector_store.vectors),
        }
        if hasattr(self.embeddings, "cache_stats"):
            stats["embedder_cache"] = self.embeddings.cache_stats
        if self._ingestion_times:
            stats["avg_ingestion_time"] = round(sum(self._ingestion_times) / len(self._ingestion_times), 2)
        return stats


class ConversationRAG:
    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine
        self.conversation_history: list[dict] = []

    def chat(self, message: str) -> str:
        retrieval = self.rag.retrieve(message)
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content'][:200]}"
            for msg in self.conversation_history[-5:]
        ])
        full_context = f"Conversation history:\n{conversation_context}\n\nRelevant documents:\n{retrieval.context}"
        response = self.rag._generate_response(
            f"Context:\n{full_context}\n\nQuestion: {message}", retrieval,
        )
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="capabilities.rag",
        description="Retrieval augmented generation for local LLMs",
    )
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--store", help="Vector store JSON file (persisted between runs)")
    parent.add_argument("--model", default=None, help="Generation model GGUF (default: Phi-4 mini in notebooks/)")
    parent.add_argument("--embedding-model", default=None, help="Embedding model GGUF (default: Qwen3-Embedding in notebooks/)")
    parent.add_argument("--cpu-percent", type=float, default=55.0, help="CPU usage cap")
    parent.add_argument("--vector-store", choices=["auto", "pgvector", "memory"], default="auto", help="Vector backend")
    parent.add_argument("--pg-dsn", default=None, help="PostgreSQL DSN for pgvector")
    parent.add_argument("--chunk-size", type=int, default=500)
    parent.add_argument("--chunk-overlap", type=int, default=50)
    parent.add_argument("--top-k", type=int, default=5)
    parent.add_argument("--embedding-dim", type=int, default=128)

    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", parents=[parent], help="Ingest a document, text, or directory")
    src = p_ing.add_mutually_exclusive_group(required=True)
    src.add_argument("--file")
    src.add_argument("--text")
    src.add_argument("--dir")
    p_ing.add_argument("--metadata", default="{}", help="JSON metadata")

    p_ret = sub.add_parser("retrieve", parents=[parent], help="Retrieve relevant chunks")
    rsrc = p_ret.add_mutually_exclusive_group(required=True)
    rsrc.add_argument("--query")
    rsrc.add_argument("--file")

    p_q = sub.add_parser("query", parents=[parent], help="Query the RAG engine")
    qsrc = p_q.add_mutually_exclusive_group(required=True)
    qsrc.add_argument("--question")
    qsrc.add_argument("--file")

    args = parser.parse_args(argv)

    def read_source(value, file_path):
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return value

    def build_engine():
        return RAGEngine(
            model_path=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            embedding_dim=args.embedding_dim,
            embedding_model=args.embedding_model,
            cpu_percent=args.cpu_percent,
            vector_store_type=args.vector_store,
            pg_dsn=args.pg_dsn,
        )

    try:
        if args.command == "ingest":
            engine = build_engine()
            meta = json.loads(args.metadata)
            if args.file:
                engine.ingest_document(args.file, metadata=meta)
            elif args.text:
                engine.ingest_text(args.text, metadata=meta)
            else:
                engine.ingest_directory(args.dir)
            if args.store:
                engine.vector_store.save(args.store)
            print(json.dumps(engine.get_stats(), indent=2, default=str))
        else:
            engine = build_engine()
            if not isinstance(engine.vector_store, PGVectorStore):
                if not args.store:
                    print("Error: --store is required for retrieve/query with in-memory store")
                    return 1
                engine.vector_store.load(args.store)
            if args.command == "retrieve":
                query = read_source(args.query, args.file)
                result = engine.retrieve(query, top_k=args.top_k)
                print(json.dumps({
                    "query": result.query,
                    "chunks": [{"id": c.id, "document_id": c.document_id, "content": c.content, "index": c.index, "metadata": c.metadata} for c in result.chunks],
                    "scores": result.scores,
                    "context": result.context,
                    "metadata": result.metadata,
                }, indent=2, default=str))
            elif args.command == "query":
                question = read_source(args.question, args.file)
                answer = engine.query(question)
                print(json.dumps({"question": question, "answer": answer}, indent=2, default=str))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

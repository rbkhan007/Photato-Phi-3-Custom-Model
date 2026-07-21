#!/usr/bin/env python3
"""
RAG (Retrieval Augmented Generation) for Local LLMs.

Features:
- Document ingestion and chunking
- Vector embeddings (local or API-based)
- Semantic search
- Context-aware generation
- Multi-document retrieval

Usage:
    from capabilities.rag import RAGEngine

    rag = RAGEngine(model_path="./phi3-mini-q4_k_m.gguf")
    rag.ingest_document("path/to/document.txt")
    response = rag.query("What is the main topic?")
"""

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# Default models live in the project's ``notebooks/`` directory.
_NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"
DEFAULT_EMBED_MODEL = str(_NOTEBOOKS / "Qwen3-Embedding-0.6B-Q8_0.gguf")
DEFAULT_GEN_MODEL = str(_NOTEBOOKS / "Phi-4-mini-instruct-Q4_K_M.gguf")


@dataclass
class Document:
    """Document representation."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    chunks: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)


@dataclass
class Chunk:
    """Document chunk."""
    id: str
    document_id: str
    content: str
    index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Retrieval result."""
    query: str
    chunks: list[Chunk]
    scores: list[float]
    context: str
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """Text chunking utilities."""

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Split text into chunks.

        Args:
            text: Input text
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            separators: Separators to split on

        Returns:
            List of text chunks
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        chunks = []
        current_chunk = ""

        # Try each separator
        for separator in separators:
            if separator == "":
                # Character-level splitting
                for i in range(0, len(text), chunk_size - chunk_overlap):
                    chunk = text[i:i + chunk_size]
                    if chunk.strip():
                        chunks.append(chunk)
                break

            # Split by separator
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

        # Ensure no empty chunks
        return [c for c in chunks if c.strip()]

    @staticmethod
    def chunk_by_sentence(text: str, max_sentences: int = 5) -> list[str]:
        """Chunk text by sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []

        for i in range(0, len(sentences), max_sentences):
            chunk = " ".join(sentences[i:i + max_sentences])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    @staticmethod
    def chunk_by_paragraph(text: str) -> list[str]:
        """Chunk text by paragraphs."""
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]


class SimpleEmbeddings:
    """Simple embedding function (TF-IDF based)."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text."""
        return re.findall(r'\b\w+\b', text.lower())

    def _tf_idf_vector(self, text: str, doc_freq: dict[str, int], total_docs: int) -> list[float]:
        """Compute TF-IDF vector."""
        tokens = self._tokenize(text)
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        vector = [0.0] * self.dimension
        for token, count in tf.items():
            # Hash token to dimension
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension
            tf_score = count / len(tokens) if tokens else 0
            idf_score = 1.0  # Simplified
            vector[idx] += tf_score * idf_score

        # Normalize
        norm = sum(v**2 for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def embed(self, text: str) -> list[float]:
        """Embed text."""
        return self._tf_idf_vector(text, {}, 1)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts."""
        return [self.embed(text) for text in texts]


class GgufEmbedder:
    """Semantic embedder backed by a local embedding GGUF via llama.cpp.

    Uses ``llama_cpp`` with ``embedding=True``. The default model is
    ``Qwen3-Embedding-0.6B-Q8_0.gguf``, which produces real dense vectors
    suitable for pgvector similarity search (replacing the toy TF-IDF
    embeddings).
    """

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
        n_embd = getattr(self._model, "n_embd", None)
        if callable(n_embd):
            try:
                n_embd = n_embd()
            except Exception:
                n_embd = None
        self.dimension = int(n_embd or 0)
        if self.dimension == 0:
            self.dimension = len(self.embed("dimension probe"))

    def embed(self, text: str) -> list[float]:
        """Embed a single text into a normalized dense vector."""
        vec = self._model.embed(text, normalize=True, truncate=True)
        return [float(x) for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        return [self.embed(t) for t in texts]


class VectorStore:
    """Simple vector store for embeddings."""

    def __init__(self):
        self.vectors: dict[str, list[float]] = {}
        self.metadata: dict[str, dict] = {}

    def add(self, id: str, vector: list[float], meta: Optional[dict] = None):
        """Add vector to store."""
        self.vectors[id] = vector
        if meta:
            self.metadata[id] = meta

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector
            top_k: Number of results
            threshold: Minimum similarity score

        Returns:
            List of (id, score) tuples
        """
        scores = []
        for id, vector in self.vectors.items():
            score = self._cosine_similarity(query_vector, vector)
            if score >= threshold:
                scores.append((id, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a**2 for a in vec1) ** 0.5
        norm2 = sum(b**2 for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def save(self, path: str):
        """Save vector store to file."""
        data = {
            "vectors": self.vectors,
            "metadata": self.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load vector store from file."""
        with open(path) as f:
            data = json.load(f)
            self.vectors = data.get("vectors", {})
            self.metadata = data.get("metadata", {})


class PGVectorStore:
    """PostgreSQL + pgvector backed vector store.

    Persists embeddings in a Postgres table and uses the ``<=>`` (cosine
    distance) operator for similarity search, replacing the in-memory
    :class:`VectorStore` when a database is available.

    The connection is resolved from ``dsn`` then the ``PGVECTOR_DSN`` /
    ``DATABASE_URL`` environment variables. Requires the ``vector`` extension,
    which is created automatically on first use. Uses ``psycopg`` (v3) when
    available, otherwise falls back to ``psycopg2``.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        table: str = "rag_vectors",
        dimension: int = 128,
    ):
        self.dsn = dsn or os.environ.get("PGVECTOR_DSN") or os.environ.get("DATABASE_URL")
        self.table = table
        self.dimension = dimension
        self.metadata: dict[str, dict] = {}
        self._conn = None
        self._driver = None

    # -- driver / connection -------------------------------------------------
    @staticmethod
    def available() -> bool:
        """Return True if a supported Postgres driver is installed."""
        try:
            import psycopg  # noqa: F401
            return True
        except Exception:
            try:
                import psycopg2  # noqa: F401
                return True
            except Exception:
                return False

    def _connect(self):
        if self._conn is not None and not self._closed():
            return self._conn
        if self.dsn is None:
            raise RuntimeError("No PostgreSQL DSN configured for PGVectorStore")
        try:
            import psycopg
            self._driver = "psycopg"
            self._conn = psycopg.connect(self.dsn)
        except Exception:
            import psycopg2
            self._driver = "psycopg2"
            self._conn = psycopg2.connect(self.dsn)
        return self._conn

    def _closed(self) -> bool:
        return getattr(self._conn, "closed", 0) not in (False, 0)

    def _cursor(self):
        return self._connect().cursor()

    def ensure_schema(self):
        """Create the pgvector extension and the vectors table if missing."""
        cur = self._cursor()
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id text PRIMARY KEY,
                    embedding vector(%s),
                    content text,
                    document_id text,
                    chunk_index integer,
                    metadata jsonb
                )
                """,
                (self.dimension,),
            )
            self._commit()
        finally:
            cur.close()

    def _commit(self):
        conn = self._connect()
        if getattr(conn, "autocommit", False):
            return
        conn.commit()

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _fmt(vec: list[float]) -> str:
        return "[" + ",".join(f"{float(x):.8g}" for x in vec) + "]"

    # -- API (mirrors VectorStore) ------------------------------------------
    def add(self, id: str, vector: list[float], meta: Optional[dict] = None):
        """Insert or upsert a single vector."""
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
            self._commit()
        finally:
            cur.close()
        self.metadata[id] = meta

    def add_batch(self, items: list[tuple[str, list[float], Optional[dict]]]):
        """Insert/upsert a batch of ``(id, vector, meta)`` tuples."""
        for item in items:
            self.add(*item)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Return ``(id, score)`` tuples ordered by cosine similarity."""
        cur = self._cursor()
        try:
            q = self._fmt(query_vector)
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
            if score >= threshold:
                results.append((rid, float(score)))
        return results[:top_k]

    @property
    def vectors(self):
        """``len(store.vectors)`` returns the number of stored embeddings."""

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

    def save(self, path: str):
        """Dump all rows to a JSON file (portable backup)."""
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
        """Restore rows previously written by :meth:`save`."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            emb = row.get("embedding")
            if emb is None:
                continue
            meta = dict(row.get("metadata") or {})
            meta.setdefault("content", row.get("content", ""))
            meta.setdefault("document_id", row.get("document_id"))
            meta.setdefault("chunk_index", row.get("chunk_index", 0))
            self.add(row["id"], emb, meta)


class RAGEngine:
    """
    RAG Engine for local LLMs.

    Features:
    - Document ingestion
    - Text chunking
    - Vector embeddings
    - Semantic search
    - Context-aware generation
    """

    def __init__(
        self,
        model_path: str = DEFAULT_GEN_MODEL,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
        embedding_dim: int = 128,
        embedding_model: str = DEFAULT_EMBED_MODEL,
        embedding_n_gpu_layers: int = 0,
        cpu_percent: float = 55.0,
        vector_store: Optional[object] = None,
        vector_store_type: str = "auto",
        pg_dsn: Optional[str] = None,
    ):
        """
        Initialize RAG engine.

        Args:
            model_path: Generation model GGUF. Answers are produced by this
                model from the retrieved context, which avoids hallucination.
            chunk_size: Chunk size for text splitting
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
            embedding_dim: Fallback embedding dimension (only used if the GGUF
                embedder is unavailable)
            embedding_model: Embedding model GGUF (default Qwen3-Embedding)
            embedding_n_gpu_layers: GPU layers to offload for the embedder
            vector_store: Explicit vector store instance
            vector_store_type: ``auto`` (pgvector if available else memory),
                ``pgvector`` (require Postgres), or ``memory`` (in-memory)
            pg_dsn: PostgreSQL DSN for pgvector store
                (also read from ``PGVECTOR_DSN`` / ``DATABASE_URL``)
        """
        self.model_path = model_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.cpu_percent = cpu_percent
        self._generator = None

        # Initialize components
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
        self.documents: dict[str, Document] = {}

    @staticmethod
    def _resolve_embedder(model_path: Optional[str], fallback_dim: int, n_gpu_layers: int, cpu_percent: float = 55.0):
        """Use the GGUF embedding model when present, else TF-IDF fallback."""
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
        """Pick the vector store backend based on ``kind`` and availability."""
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

    def ingest_document(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
    ) -> Document:
        """
        Ingest a document.

        Args:
            file_path: Path to document
            metadata: Additional metadata

        Returns:
            Document
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read content
        content = path.read_text(encoding="utf-8")

        # Create document
        doc_id = hashlib.md5(content.encode()).hexdigest()
        doc = Document(
            id=doc_id,
            content=content,
            metadata={
                "source": str(path.absolute()),
                "filename": path.name,
                "size": len(content),
                **(metadata or {}),
            },
        )

        # Chunk document
        doc.chunks = self.chunker.chunk_text(
            content,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Create embeddings and store
        for i, chunk_text in enumerate(doc.chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self.embeddings.embed(chunk_text)
            self.vector_store.add(
                chunk_id,
                embedding,
                meta={
                    "document_id": doc_id,
                    "chunk_index": i,
                    "content": chunk_text,
                },
            )

        self.documents[doc_id] = doc
        print(f"Ingested document: {path.name} ({len(doc.chunks)} chunks)")

        return doc

    def ingest_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> Document:
        """
        Ingest raw text.

        Args:
            text: Text content
            metadata: Additional metadata

        Returns:
            Document
        """
        doc_id = hashlib.md5(text.encode()).hexdigest()
        doc = Document(
            id=doc_id,
            content=text,
            metadata=metadata or {},
        )

        doc.chunks = self.chunker.chunk_text(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        for i, chunk_text in enumerate(doc.chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self.embeddings.embed(chunk_text)
            self.vector_store.add(
                chunk_id,
                embedding,
                meta={
                    "document_id": doc_id,
                    "chunk_index": i,
                    "content": chunk_text,
                },
            )

        self.documents[doc_id] = doc
        print(f"Ingested text ({len(doc.chunks)} chunks)")

        return doc

    def ingest_directory(
        self,
        dir_path: str,
        extensions: Optional[list[str]] = None,
    ) -> list[Document]:
        """
        Ingest all documents in a directory.

        Args:
            dir_path: Directory path
            extensions: File extensions to include

        Returns:
            List of ingested documents
        """
        if extensions is None:
            extensions = [".txt", ".md", ".py", ".json"]

        documents = []
        path = Path(dir_path)

        for file_path in path.rglob("*"):
            if file_path.suffix in extensions:
                try:
                    doc = self.ingest_document(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    print(f"Error ingesting {file_path}: {e}")

        return documents

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            RetrievalResult
        """
        top_k = top_k or self.top_k

        # Embed query
        query_embedding = self.embeddings.embed(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # Build chunks and scores
        chunks = []
        scores = []
        for chunk_id, score in results:
            meta = self.vector_store.metadata.get(chunk_id, {})
            chunk = Chunk(
                id=chunk_id,
                document_id=meta.get("document_id", ""),
                content=meta.get("content", ""),
                index=meta.get("chunk_index", 0),
                metadata=meta,
            )
            chunks.append(chunk)
            scores.append(score)

        # Build context
        context = "\n\n---\n\n".join([c.content for c in chunks])

        return RetrievalResult(
            query=query,
            chunks=chunks,
            scores=scores,
            context=context,
            metadata={"num_chunks": len(chunks)},
        )

    def query(
        self,
        question: str,
        context_template: Optional[str] = None,
    ) -> str:
        """
        Query the RAG system.

        Args:
            question: User question
            context_template: Template for context

        Returns:
            Response string
        """
        # Retrieve relevant chunks
        retrieval = self.retrieve(question)

        # Build prompt with context
        if context_template is None:
            context_template = """Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer based on the context above. If the context doesn't contain enough information, say so."""

        prompt = context_template.format(
            context=retrieval.context,
            question=question,
        )

        # Generate response (placeholder - use actual model)
        response = self._generate_response(prompt, retrieval)

        return response

    def _generate_response(self, prompt: str, retrieval: RetrievalResult) -> str:
        """
        Generate a response grounded in the retrieved context.

        Uses the real generation model (loaded lazily from ``model_path``) so
        the answer is conditioned on the retrieved chunks instead of being a
        hardcoded placeholder. Falls back to a context-only stub only if the
        model cannot be loaded.

        Args:
            prompt: Full prompt with context
            retrieval: Retrieval result

        Returns:
            Generated response text
        """
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
                    prompt,
                    max_tokens=512,
                    temperature=0.1,
                    repeat_penalty=1.15,
                )
                text = result.get("text", "") if isinstance(result, dict) else str(result)
                return text.strip()
            except Exception as e:
                print(f"[rag] generation failed ({e}); returning context-only stub")
                self._generator = False
        return (
            f"Based on the retrieved context ({len(retrieval.chunks)} chunks), "
            "here is the answer..."
        )

    def get_stats(self) -> dict:
        """Get RAG engine statistics."""
        return {
            "documents": len(self.documents),
            "total_chunks": sum(len(d.chunks) for d in self.documents.values()),
            "vector_store_size": len(self.vector_store.vectors),
        }


class ConversationRAG:
    """RAG with conversation memory."""

    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine
        self.conversation_history: list[dict] = []

    def chat(self, message: str) -> str:
        """
        Chat with RAG.

        Args:
            message: User message

        Returns:
            Response
        """
        # Retrieve relevant context
        retrieval = self.rag.retrieve(message)

        # Build conversation context
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content'][:200]}"
            for msg in self.conversation_history[-5:]
        ])

        # Combine contexts
        full_context = f"Conversation history:\n{conversation_context}\n\nRelevant documents:\n{retrieval.context}"

        # Generate response
        response = self.rag._generate_response(
            f"Context:\n{full_context}\n\nQuestion: {message}",
            retrieval,
        )

        # Update history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response


def main(argv=None):
    """CLI for retrieval augmented generation."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="capabilities.rag",
        description="Retrieval augmented generation for local LLMs",
    )
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--store", help="Vector store JSON file (persisted between runs)")
    parent.add_argument(
        "--model",
        default=None,
        help="Generation model GGUF (default: Phi-4 mini in notebooks/)",
    )
    parent.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model GGUF (default: Qwen3-Embedding in notebooks/)",
    )
    parent.add_argument(
        "--cpu-percent",
        type=float,
        default=55.0,
        help="Cap CPU usage to this percent (Windows Job Object hard cap; 100=unlimited).",
    )
    parent.add_argument(
        "--vector-store",
        choices=["auto", "pgvector", "memory"],
        default="auto",
        help="Vector backend: auto (pgvector if available), pgvector, or memory",
    )
    parent.add_argument(
        "--pg-dsn",
        default=None,
        help="PostgreSQL DSN for pgvector (else PGVECTOR_DSN / DATABASE_URL env)",
    )
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
    p_ing.add_argument("--metadata", default="{}", help="JSON metadata for ingestion")

    p_ret = sub.add_parser("retrieve", parents=[parent], help="Retrieve relevant chunks for a query")
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

    def chunk_to_dict(c):
        return {
            "id": c.id,
            "document_id": c.document_id,
            "content": c.content,
            "index": c.index,
            "metadata": c.metadata,
        }

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
                    print("Error: --store is required for retrieve/query with the in-memory store", file=sys.stderr)
                    return 1
                engine.vector_store.load(args.store)
            if args.command == "retrieve":
                query = read_source(args.query, args.file)
                result = engine.retrieve(query, top_k=args.top_k)
                print(json.dumps({
                    "query": result.query,
                    "chunks": [chunk_to_dict(c) for c in result.chunks],
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
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

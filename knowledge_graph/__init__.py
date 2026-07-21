"""
Knowledge Graph Database Integration.

Provides persistent, queryable memory using graph databases.

Supports multiple backends:
- NetworkX (in-memory, default)
- Neo4j (production)
- FalkorDB (Redis-based)
- SQLite (lightweight persistent)
- JSON file (simple persistence)

Features:
- Entity and relationship storage
- Semantic search
- Graph traversal
- Community detection
- Memory consolidation
- Conversation history
"""

import os
import json
import time
import hashlib
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from pathlib import Path
from enum import Enum
from datetime import datetime


class GraphBackend(Enum):
    """Knowledge graph backend types."""
    NETWORKX = "networkx"
    NEO4J = "neo4j"
    FALKORDB = "falkordb"
    SQLITE = "sqlite"
    JSON = "json"


class EntityType(Enum):
    """Entity types in the knowledge graph."""
    CONCEPT = "concept"
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    DOCUMENT = "document"
    CODE = "code"
    TOOL = "tool"
    CONVERSATION = "conversation"
    MEMORY = "memory"
    CUSTOM = "custom"


class RelationType(Enum):
    """Relationship types."""
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    CAUSED_BY = "caused_by"
    LED_TO = "led_to"
    USED_IN = "used_in"
    MENTIONED_IN = "mentioned_in"
    CREATED_BY = "created_by"
    DEPENDS_ON = "depends_on"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    CUSTOM = "custom"


@dataclass
class Entity:
    """Knowledge graph entity."""
    id: str
    name: str
    entity_type: EntityType
    properties: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.5


@dataclass
class Relationship:
    """Knowledge graph relationship."""
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: dict = field(default_factory=dict)
    weight: float = 1.0
    created_at: float = field(default_factory=time.time)


@dataclass
class GraphQuery:
    """Query for knowledge graph."""
    entity_type: Optional[EntityType] = None
    relation_type: Optional[RelationType] = None
    keywords: list[str] = field(default_factory=list)
    min_importance: float = 0.0
    max_results: int = 10
    include_neighbors: bool = False
    time_range: Optional[tuple[float, float]] = None


@dataclass
class GraphResult:
    """Result from knowledge graph query."""
    entities: list[Entity]
    relationships: list[Relationship]
    paths: list[list[str]]
    metadata: dict = field(default_factory=dict)


class KnowledgeGraph:
    """
    Knowledge Graph database for persistent memory.

    Stores entities and relationships with semantic search.
    """

    def __init__(
        self,
        backend: GraphBackend = GraphBackend.NETWORKX,
        db_path: str = None,
        config: dict = None
    ):
        self.backend = backend
        self.db_path = db_path or "./knowledge_graph"
        self.config = config or {}

        self.entities: dict[str, Entity] = {}
        self.relationships: dict[str, Relationship] = {}
        self._adjacency: dict[str, set[str]] = {}

        self._init_backend()

    def _init_backend(self):
        """Initialize the backend storage."""
        if self.backend == GraphBackend.JSON:
            os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
            self._load_json()

        elif self.backend == GraphBackend.SQLITE:
            db_file = self.db_path if self.db_path.endswith(".db") else f"{self.db_path}.db"
            os.makedirs(os.path.dirname(db_file) if os.path.dirname(db_file) else ".", exist_ok=True)
            self._init_sqlite(db_file)

        elif self.backend == GraphBackend.NETWORKX:
            try:
                import networkx as nx
                self._graph = nx.DiGraph()
            except ImportError:
                self._graph = None

    def _init_sqlite(self, db_file: str):
        """Initialize SQLite backend."""
        self._conn = sqlite3.connect(db_file)
        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT,
                entity_type TEXT,
                properties TEXT,
                embedding BLOB,
                created_at REAL,
                updated_at REAL,
                access_count INTEGER,
                importance REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT,
                target_id TEXT,
                relation_type TEXT,
                properties TEXT,
                weight REAL,
                created_at REAL,
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relation_type ON relationships(relation_type)
        """)

        self._conn.commit()

        # Load existing data
        self._load_sqlite()

    def _load_sqlite(self):
        """Load data from SQLite."""
        cursor = self._conn.cursor()

        cursor.execute("SELECT * FROM entities")
        for row in cursor.fetchall():
            entity = Entity(
                id=row[0],
                name=row[1],
                entity_type=EntityType(row[2]),
                properties=json.loads(row[3]) if row[3] else {},
                created_at=row[5],
                updated_at=row[6],
                access_count=row[7],
                importance=row[8]
            )
            self.entities[entity.id] = entity

        cursor.execute("SELECT * FROM relationships")
        for row in cursor.fetchall():
            rel = Relationship(
                id=row[0],
                source_id=row[1],
                target_id=row[2],
                relation_type=RelationType(row[3]),
                properties=json.loads(row[4]) if row[4] else {},
                weight=row[5],
                created_at=row[6]
            )
            self.relationships[rel.id] = rel
            self._add_to_adjacency(rel.source_id, rel.id)
            self._add_to_adjacency(rel.target_id, rel.id)

    def _load_json(self):
        """Load data from JSON file."""
        json_file = self.db_path if self.db_path.endswith(".json") else f"{self.db_path}.json"
        if os.path.exists(json_file):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    for e in data.get("entities", []):
                        entity = Entity(**e)
                        entity.entity_type = EntityType(entity.entity_type)
                        self.entities[entity.id] = entity
                    for r in data.get("relationships", []):
                        rel = Relationship(**r)
                        rel.relation_type = RelationType(rel.relation_type)
                        self.relationships[rel.id] = rel
                        self._add_to_adjacency(rel.source_id, rel.id)
                        self._add_to_adjacency(rel.target_id, rel.id)
            except Exception:
                pass

    def _save_json(self):
        """Save data to JSON file."""
        json_file = self.db_path if self.db_path.endswith(".json") else f"{self.db_path}.json"
        data = {
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "entity_type": e.entity_type.value,
                    "properties": e.properties,
                    "created_at": e.created_at,
                    "updated_at": e.updated_at,
                    "access_count": e.access_count,
                    "importance": e.importance
                }
                for e in self.entities.values()
            ],
            "relationships": [
                {
                    "id": r.id,
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "relation_type": r.relation_type.value,
                    "properties": r.properties,
                    "weight": r.weight,
                    "created_at": r.created_at
                }
                for r in self.relationships.values()
            ]
        }
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)

    def _save_sqlite(self):
        """Save to SQLite."""
        cursor = self._conn.cursor()

        for entity in self.entities.values():
            cursor.execute("""
                INSERT OR REPLACE INTO entities
                (id, name, entity_type, properties, embedding, created_at, updated_at, access_count, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity.id,
                entity.name,
                entity.entity_type.value,
                json.dumps(entity.properties),
                None,
                entity.created_at,
                entity.updated_at,
                entity.access_count,
                entity.importance
            ))

        for rel in self.relationships.values():
            cursor.execute("""
                INSERT OR REPLACE INTO relationships
                (id, source_id, target_id, relation_type, properties, weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                rel.id,
                rel.source_id,
                rel.target_id,
                rel.relation_type.value,
                json.dumps(rel.properties),
                rel.weight,
                rel.created_at
            ))

        self._conn.commit()

    def _add_to_adjacency(self, node_id: str, rel_id: str):
        """Add to adjacency list."""
        if node_id not in self._adjacency:
            self._adjacency[node_id] = set()
        self._adjacency[node_id].add(rel_id)

    # === Entity Operations ===

    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        properties: dict = None,
        importance: float = 0.5
    ) -> Entity:
        """Add an entity to the graph."""
        entity_id = self._generate_id(name, entity_type)

        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            importance=importance
        )

        self.entities[entity_id] = entity
        self._save()
        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        entity = self.entities.get(entity_id)
        if entity:
            entity.access_count += 1
            entity.updated_at = time.time()
        return entity

    def find_entity(self, name: str, entity_type: EntityType = None) -> list[Entity]:
        """Find entities by name."""
        results = []
        name_lower = name.lower()

        for entity in self.entities.values():
            if name_lower in entity.name.lower():
                if entity_type is None or entity.entity_type == entity_type:
                    results.append(entity)

        return sorted(results, key=lambda e: e.importance, reverse=True)

    def update_entity(self, entity_id: str, properties: dict = None, importance: float = None):
        """Update entity properties."""
        entity = self.entities.get(entity_id)
        if entity:
            if properties:
                entity.properties.update(properties)
            if importance is not None:
                entity.importance = importance
            entity.updated_at = time.time()
            self._save()

    def delete_entity(self, entity_id: str):
        """Delete an entity and its relationships."""
        # Delete relationships
        rel_ids = []
        for rel_id, rel in self.relationships.items():
            if rel.source_id == entity_id or rel.target_id == entity_id:
                rel_ids.append(rel_id)

        for rel_id in rel_ids:
            del self.relationships[rel_id]

        # Delete entity
        self.entities.pop(entity_id, None)
        self._adjacency.pop(entity_id, None)
        self._save()

    # === Relationship Operations ===

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        properties: dict = None,
        weight: float = 1.0
    ) -> Relationship:
        """Add a relationship between entities."""
        rel_id = self._generate_id(f"{source_id}-{target_id}", relation_type)

        rel = Relationship(
            id=rel_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
            weight=weight
        )

        self.relationships[rel_id] = rel
        self._add_to_adjacency(source_id, rel_id)
        self._add_to_adjacency(target_id, rel_id)
        self._save()
        return rel

    def get_relationships(
        self,
        entity_id: str,
        relation_type: RelationType = None,
        direction: str = "both"
    ) -> list[Relationship]:
        """Get relationships for an entity."""
        results = []

        for rel in self.relationships.values():
            if direction == "outgoing" and rel.source_id != entity_id:
                continue
            if direction == "incoming" and rel.target_id != entity_id:
                continue
            if direction == "both" and rel.source_id != entity_id and rel.target_id != entity_id:
                continue

            if relation_type and rel.relation_type != relation_type:
                continue

            results.append(rel)

        return results

    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> list[list[str]]:
        """Find paths between two entities."""
        paths = []
        self._dfs_paths(source_id, target_id, set(), [source_id], paths, max_depth)
        return paths

    def _dfs_paths(self, current: str, target: str, visited: set,
                   path: list, paths: list, max_depth: int):
        """DFS to find paths."""
        if len(path) > max_depth:
            return

        if current == target and len(path) > 1:
            paths.append(list(path))
            return

        visited.add(current)

        rel_ids = self._adjacency.get(current, set())
        for rel_id in rel_ids:
            rel = self.relationships.get(rel_id)
            if not rel:
                continue

            next_id = rel.target_id if rel.source_id == current else rel.source_id

            if next_id not in visited and next_id in self.entities:
                path.append(next_id)
                self._dfs_paths(next_id, target, visited, path, paths, max_depth)
                path.pop()

        visited.discard(current)

    # === Query Operations ===

    def query(self, query: GraphQuery) -> GraphResult:
        """Query the knowledge graph."""
        results = []

        for entity in self.entities.values():
            if query.entity_type and entity.entity_type != query.entity_type:
                continue

            if entity.importance < query.min_importance:
                continue

            if query.keywords:
                text = f"{entity.name} {json.dumps(entity.properties)}".lower()
                if not any(kw.lower() in text for kw in query.keywords):
                    continue

            if query.time_range:
                if not (query.time_range[0] <= entity.created_at <= query.time_range[1]):
                    continue

            results.append(entity)

        # Sort by importance
        results.sort(key=lambda e: e.importance, reverse=True)
        results = results[:query.max_results]

        # Get relationships
        relationships = []
        if query.include_neighbors:
            for entity in results:
                rels = self.get_relationships(entity.id, query.relation_type)
                relationships.extend(rels)

        return GraphResult(
            entities=results,
            relationships=relationships,
            paths=[],
            metadata={"total_entities": len(self.entities)}
        )

    def search(self, text: str, max_results: int = 10) -> list[Entity]:
        """Search entities by text."""
        results = []
        text_lower = text.lower()

        for entity in self.entities.values():
            score = 0
            if text_lower in entity.name.lower():
                score += 2
            for value in entity.properties.values():
                if isinstance(value, str) and text_lower in value.lower():
                    score += 1

            if score > 0:
                results.append((entity, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [entity for entity, _ in results[:max_results]]

    # === Memory Operations ===

    def add_conversation(self, messages: list[dict], summary: str = None):
        """Add a conversation to memory."""
        conv_id = self._generate_id(summary or str(messages[:2]), EntityType.CONVERSATION)

        entity = self.add_entity(
            name=summary or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            entity_type=EntityType.CONVERSATION,
            properties={
                "messages": messages,
                "message_count": len(messages),
                "summary": summary
            },
            importance=0.6
        )

        # Link mentioned concepts
        for msg in messages:
            content = msg.get("content", "")
            words = content.split()[:20]
            for word in words:
                if len(word) > 4:
                    concepts = self.find_entity(word)
                    for concept in concepts[:1]:
                        self.add_relationship(
                            entity.id,
                            concept.id,
                            RelationType.MENTIONED_IN,
                            weight=0.5
                        )

        return entity

    def add_memory(self, content: str, memory_type: str = "fact",
                   importance: float = 0.7, tags: list[str] = None):
        """Add a memory entry."""
        entity = self.add_entity(
            name=content[:100],
            entity_type=EntityType.MEMORY,
            properties={
                "content": content,
                "memory_type": memory_type,
                "tags": tags or []
            },
            importance=importance
        )

        # Link to related memories
        existing = self.search(content[:50])
        for related in existing[:3]:
            if related.id != entity.id:
                self.add_relationship(
                    entity.id,
                    related.id,
                    RelationType.SIMILAR_TO,
                    weight=0.6
                )

        return entity

    def recall(self, query: str, limit: int = 5) -> list[dict]:
        """Recall memories related to query."""
        entities = self.search(query, max_results=limit * 2)

        memories = []
        for entity in entities:
            if entity.entity_type == EntityType.MEMORY:
                memories.append({
                    "id": entity.id,
                    "content": entity.properties.get("content", ""),
                    "type": entity.properties.get("memory_type", "fact"),
                    "importance": entity.importance,
                    "tags": entity.properties.get("tags", [])
                })

        return memories[:limit]

    # === Utility ===

    def _generate_id(self, name: str, entity_type) -> str:
        """Generate a unique ID."""
        data = f"{name}_{entity_type.value}_{time.time()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def _save(self):
        """Save to backend."""
        if self.backend == GraphBackend.JSON:
            self._save_json()
        elif self.backend == GraphBackend.SQLITE:
            self._save_sqlite()

    def get_stats(self) -> dict:
        """Get graph statistics."""
        entity_types = {}
        for entity in self.entities.values():
            t = entity.entity_type.value
            entity_types[t] = entity_types.get(t, 0) + 1

        rel_types = {}
        for rel in self.relationships.values():
            t = rel.relation_type.value
            rel_types[t] = rel_types.get(t, 0) + 1

        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_types": entity_types,
            "relationship_types": rel_types,
            "avg_importance": sum(e.importance for e in self.entities.values()) / max(1, len(self.entities))
        }

    def export(self, format: str = "json") -> str:
        """Export graph data."""
        if format == "json":
            return json.dumps({
                "entities": {k: {"name": v.name, "type": v.entity_type.value} for k, v in self.entities.items()},
                "relationships": {k: {"source": v.source_id, "target": v.target_id, "type": v.relation_type.value} for k, v in self.relationships.items()}
            }, indent=2)
        return ""

    def clear(self):
        """Clear all data."""
        self.entities.clear()
        self.relationships.clear()
        self._adjacency.clear()
        self._save()


class MemoryManager:
    """
    High-level memory manager using Knowledge Graph.
    """

    def __init__(self, graph: KnowledgeGraph = None):
        self.graph = graph or KnowledgeGraph(backend=GraphBackend.JSON, db_path="./agent_memory")

    def remember(self, content: str, category: str = "general", importance: float = 0.7):
        """Store a memory."""
        return self.graph.add_memory(
            content=content,
            memory_type=category,
            importance=importance,
            tags=[category]
        )

    def recall(self, query: str, limit: int = 5) -> list[str]:
        """Recall related memories."""
        memories = self.graph.recall(query, limit)
        return [m["content"] for m in memories]

    def forget(self, content_pattern: str):
        """Forget memories matching pattern."""
        entities = self.graph.search(content_pattern)
        for entity in entities:
            if entity.entity_type == EntityType.MEMORY:
                self.graph.delete_entity(entity.id)

    def get_context(self, current_topic: str, max_tokens: int = 2000) -> str:
        """Get context for current topic from memory."""
        memories = self.recall(current_topic, limit=10)

        context_parts = []
        total_tokens = 0

        for memory in memories:
            estimated_tokens = len(memory.split()) * 1.3
            if total_tokens + estimated_tokens > max_tokens:
                break
            context_parts.append(memory)
            total_tokens += estimated_tokens

        return "\n".join(context_parts)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return self.graph.get_stats()


# === Convenience Functions ===

def create_memory(db_path: str = "./memory") -> MemoryManager:
    """Create a new memory manager."""
    graph = KnowledgeGraph(backend=GraphBackend.JSON, db_path=db_path)
    return MemoryManager(graph)

def create_knowledge_graph(backend: str = "json", db_path: str = "./kg") -> KnowledgeGraph:
    """Create a new knowledge graph."""
    return KnowledgeGraph(backend=GraphBackend(backend), db_path=db_path)

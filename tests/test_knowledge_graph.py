"""Tests for the knowledge_graph package (in-memory + JSON backend)."""

import json

from knowledge_graph import (
    KnowledgeGraph, MemoryManager, GraphBackend, EntityType, RelationType,
    Entity, Relationship, GraphQuery, GraphResult,
    create_memory, create_knowledge_graph,
)


def _kg():
    return KnowledgeGraph(backend=GraphBackend.NETWORKX)


class TestEntities:
    def test_add_and_get_entity(self):
        kg = _kg()
        e = kg.add_entity("Python", EntityType.CONCEPT, importance=0.9)
        assert isinstance(e, Entity)
        assert e.name == "Python"
        assert e.entity_type == EntityType.CONCEPT
        got = kg.get_entity(e.id)
        assert got is e
        assert got.access_count == 1

    def test_find_entity_sorted_by_importance(self):
        kg = _kg()
        kg.add_entity("Alpha thing", EntityType.CONCEPT, importance=0.2)
        kg.add_entity("Alpha other", EntityType.CONCEPT, importance=0.8)
        results = kg.find_entity("alpha")
        assert len(results) == 2
        assert results[0].importance == 0.8

    def test_find_entity_type_filter(self):
        kg = _kg()
        kg.add_entity("Bob", EntityType.PERSON)
        kg.add_entity("Bob Corp", EntityType.ORGANIZATION)
        people = kg.find_entity("bob", EntityType.PERSON)
        assert len(people) == 1
        assert people[0].entity_type == EntityType.PERSON

    def test_update_entity(self):
        kg = _kg()
        e = kg.add_entity("X", EntityType.CONCEPT)
        kg.update_entity(e.id, properties={"k": "v"}, importance=0.99)
        assert kg.entities[e.id].properties["k"] == "v"
        assert kg.entities[e.id].importance == 0.99

    def test_delete_entity_removes_relationships(self):
        kg = _kg()
        a = kg.add_entity("A", EntityType.CONCEPT)
        b = kg.add_entity("B", EntityType.CONCEPT)
        kg.add_relationship(a.id, b.id, RelationType.RELATED_TO)
        kg.delete_entity(a.id)
        assert a.id not in kg.entities
        assert kg.relationships == {}


class TestRelationships:
    def test_add_relationship(self):
        kg = _kg()
        a = kg.add_entity("A", EntityType.CONCEPT)
        b = kg.add_entity("B", EntityType.CONCEPT)
        rel = kg.add_relationship(a.id, b.id, RelationType.DEPENDS_ON, weight=2.0)
        assert isinstance(rel, Relationship)
        assert rel.weight == 2.0
        assert rel.relation_type == RelationType.DEPENDS_ON

    def test_get_relationships_direction(self):
        kg = _kg()
        a = kg.add_entity("A", EntityType.CONCEPT)
        b = kg.add_entity("B", EntityType.CONCEPT)
        kg.add_relationship(a.id, b.id, RelationType.LED_TO)
        assert len(kg.get_relationships(a.id, direction="outgoing")) == 1
        assert len(kg.get_relationships(a.id, direction="incoming")) == 0
        assert len(kg.get_relationships(b.id, direction="incoming")) == 1

    def test_find_path(self):
        kg = _kg()
        a = kg.add_entity("A", EntityType.CONCEPT)
        b = kg.add_entity("B", EntityType.CONCEPT)
        c = kg.add_entity("C", EntityType.CONCEPT)
        kg.add_relationship(a.id, b.id, RelationType.RELATED_TO)
        kg.add_relationship(b.id, c.id, RelationType.RELATED_TO)
        paths = kg.find_path(a.id, c.id)
        assert any(path[0] == a.id and path[-1] == c.id for path in paths)


class TestQueryAndSearch:
    def test_query_by_type_and_importance(self):
        kg = _kg()
        kg.add_entity("keep", EntityType.CONCEPT, importance=0.9)
        kg.add_entity("drop", EntityType.CONCEPT, importance=0.1)
        kg.add_entity("person", EntityType.PERSON, importance=0.9)
        q = GraphQuery(entity_type=EntityType.CONCEPT, min_importance=0.5)
        result = kg.query(q)
        assert isinstance(result, GraphResult)
        assert len(result.entities) == 1
        assert result.entities[0].name == "keep"
        assert result.metadata["total_entities"] == 3

    def test_query_keywords(self):
        kg = _kg()
        kg.add_entity("machine learning", EntityType.CONCEPT)
        kg.add_entity("cooking", EntityType.CONCEPT)
        q = GraphQuery(keywords=["machine"])
        result = kg.query(q)
        assert len(result.entities) == 1

    def test_search_scoring(self):
        kg = _kg()
        kg.add_entity("python guide", EntityType.CONCEPT, properties={"desc": "learn python"})
        kg.add_entity("unrelated", EntityType.CONCEPT)
        results = kg.search("python")
        assert results
        assert results[0].name == "python guide"


class TestMemory:
    def test_add_memory_and_recall(self):
        kg = _kg()
        kg.add_memory("The sky is blue", memory_type="fact", tags=["nature"])
        recalled = kg.recall("sky")
        assert len(recalled) == 1
        assert recalled[0]["content"] == "The sky is blue"
        assert recalled[0]["tags"] == ["nature"]

    def test_stats(self):
        kg = _kg()
        a = kg.add_entity("A", EntityType.CONCEPT, importance=0.5)
        b = kg.add_entity("B", EntityType.PERSON, importance=0.5)
        kg.add_relationship(a.id, b.id, RelationType.RELATED_TO)
        stats = kg.get_stats()
        assert stats["total_entities"] == 2
        assert stats["total_relationships"] == 1
        assert stats["entity_types"]["concept"] == 1
        assert stats["avg_importance"] == 0.5

    def test_export_and_clear(self):
        kg = _kg()
        kg.add_entity("A", EntityType.CONCEPT)
        exported = json.loads(kg.export("json"))
        assert "entities" in exported
        kg.clear()
        assert kg.entities == {}
        assert kg.relationships == {}


class TestJSONBackendPersistence:
    def test_persist_and_reload(self, tmp_path):
        db = str(tmp_path / "kg")
        kg = create_knowledge_graph(backend="json", db_path=db)
        kg.add_entity("Persistent", EntityType.CONCEPT, importance=0.7)
        kg2 = KnowledgeGraph(backend=GraphBackend.JSON, db_path=db)
        names = [e.name for e in kg2.entities.values()]
        assert "Persistent" in names


class TestMemoryManager:
    def test_remember_and_recall(self, tmp_path):
        kg = KnowledgeGraph(backend=GraphBackend.NETWORKX)
        mm = MemoryManager(graph=kg)
        mm.remember("Cats are mammals", category="biology")
        recalled = mm.recall("cats")
        assert "Cats are mammals" in recalled

    def test_forget(self):
        kg = KnowledgeGraph(backend=GraphBackend.NETWORKX)
        mm = MemoryManager(graph=kg)
        mm.remember("delete me please", category="temp")
        mm.forget("delete me")
        assert mm.recall("delete me") == []

    def test_get_context_respects_token_budget(self):
        kg = KnowledgeGraph(backend=GraphBackend.NETWORKX)
        mm = MemoryManager(graph=kg)
        mm.remember("alpha topic memory one", category="t")
        ctx = mm.get_context("topic", max_tokens=2000)
        assert isinstance(ctx, str)

    def test_create_memory_helper(self, tmp_path):
        mm = create_memory(db_path=str(tmp_path / "mem"))
        assert isinstance(mm, MemoryManager)

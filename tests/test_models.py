"""Tests for core data models."""

import pytest
from cgqa.models.graph import Entity, Relationship, KnowledgeGraph
from cgqa.models.question import Question, QuestionType, Answer, AnswerType


class TestEntity:
    """Test Entity model."""
    
    def test_entity_creation(self):
        """Test basic entity creation."""
        entity = Entity(
            id="test_entity_1",
            name="Test Entity",
            entity_type="person"
        )
        
        assert entity.id == "test_entity_1"
        assert entity.name == "Test Entity"
        assert entity.entity_type == "person"
        assert entity.properties == {}
    
    def test_entity_with_properties(self):
        """Test entity with custom properties."""
        entity = Entity(
            id="test_entity_2",
            name="Test Entity 2",
            entity_type="place",
            properties={"population": 100000, "country": "TestLand"}
        )
        
        assert entity.properties["population"] == 100000
        assert entity.properties["country"] == "TestLand"
    
    def test_entity_equality(self):
        """Test entity equality based on ID."""
        entity1 = Entity(id="same_id", name="Entity 1")
        entity2 = Entity(id="same_id", name="Entity 2")  # Different name, same ID
        entity3 = Entity(id="different_id", name="Entity 1")
        
        assert entity1 == entity2  # Same ID
        assert entity1 != entity3  # Different ID
    
    def test_entity_hash(self):
        """Test entity hashing."""
        entity1 = Entity(id="test_id", name="Test")
        entity2 = Entity(id="test_id", name="Test")
        
        assert hash(entity1) == hash(entity2)
        assert {entity1, entity2} == {entity1}  # Should be treated as same in set


class TestRelationship:
    """Test Relationship model."""
    
    def test_relationship_creation(self):
        """Test basic relationship creation."""
        rel = Relationship(
            source="entity_1",
            target="entity_2",
            relation_type="knows"
        )
        
        assert rel.source == "entity_1"
        assert rel.target == "entity_2"
        assert rel.relation_type == "knows"
        assert rel.weight is None
        assert rel.properties == {}
    
    def test_relationship_with_weight(self):
        """Test relationship with weight."""
        rel = Relationship(
            source="entity_1",
            target="entity_2",
            relation_type="friend_of",
            weight=0.8
        )
        
        assert rel.weight == 0.8
    
    def test_relationship_equality(self):
        """Test relationship equality."""
        rel1 = Relationship(source="a", target="b", relation_type="knows")
        rel2 = Relationship(source="a", target="b", relation_type="knows")
        rel3 = Relationship(source="a", target="c", relation_type="knows")
        
        assert rel1 == rel2
        assert rel1 != rel3


class TestKnowledgeGraph:
    """Test KnowledgeGraph model."""
    
    def test_empty_graph(self):
        """Test empty knowledge graph."""
        kg = KnowledgeGraph()
        
        assert len(kg.entities) == 0
        assert len(kg.relationships) == 0
        assert kg.metadata == {}
    
    def test_add_entity(self):
        """Test adding entities to graph."""
        kg = KnowledgeGraph()
        entity = Entity(id="test_1", name="Test Entity")
        
        kg.add_entity(entity)
        
        assert len(kg.entities) == 1
        assert kg.entities["test_1"] == entity
        assert kg.get_entity("test_1") == entity
    
    def test_add_relationship(self):
        """Test adding relationships to graph."""
        kg = KnowledgeGraph()
        
        # Add entities first
        entity1 = Entity(id="e1", name="Entity 1")
        entity2 = Entity(id="e2", name="Entity 2")
        kg.add_entity(entity1)
        kg.add_entity(entity2)
        
        # Add relationship
        rel = Relationship(source="e1", target="e2", relation_type="knows")
        kg.add_relationship(rel)
        
        assert len(kg.relationships) == 1
        assert kg.relationships[0] == rel
    
    def test_add_relationship_missing_entities(self):
        """Test that adding relationship with missing entities raises error."""
        kg = KnowledgeGraph()
        rel = Relationship(source="missing_1", target="missing_2", relation_type="knows")
        
        with pytest.raises(ValueError):
            kg.add_relationship(rel)
    
    def test_get_neighbors(self):
        """Test getting neighboring entities."""
        kg = KnowledgeGraph()
        
        # Create entities
        for i in range(3):
            entity = Entity(id=f"e{i}", name=f"Entity {i}")
            kg.add_entity(entity)
        
        # Add relationships: e0 -> e1 -> e2
        rel1 = Relationship(source="e0", target="e1", relation_type="knows")
        rel2 = Relationship(source="e1", target="e2", relation_type="friend")
        kg.add_relationship(rel1)
        kg.add_relationship(rel2)
        
        # Test neighbors
        neighbors_e0 = kg.get_neighbors("e0")
        neighbors_e1 = kg.get_neighbors("e1")
        
        assert neighbors_e0 == {"e1"}
        assert neighbors_e1 == {"e0", "e2"}
    
    def test_graph_stats(self):
        """Test graph statistics."""
        kg = KnowledgeGraph()
        
        # Add some entities and relationships
        entities = [Entity(id=f"e{i}", name=f"Entity {i}", entity_type="person") for i in range(3)]
        for entity in entities:
            kg.add_entity(entity)
        
        rel = Relationship(source="e0", target="e1", relation_type="knows")
        kg.add_relationship(rel)
        
        stats = kg.get_stats()
        
        assert stats["num_entities"] == 3
        assert stats["num_relationships"] == 1
        assert "person" in stats["entity_types"]
        assert "knows" in stats["relation_types"]


class TestQuestion:
    """Test Question model."""
    
    def test_question_creation(self):
        """Test basic question creation."""
        answer = Answer(value=True, answer_type=AnswerType.BOOLEAN)
        question = Question(
            id="q1",
            question_text="Is there a path between A and B?",
            question_type=QuestionType.MULTIHOP,
            complexity_level=1,
            ground_truth=answer
        )
        
        assert question.id == "q1"
        assert question.question_type == QuestionType.MULTIHOP
        assert question.complexity_level == 1
        assert question.ground_truth.value is True
        assert question.is_correct is None  # Not evaluated yet
    
    def test_answer_types(self):
        """Test different answer types."""
        # Boolean answer
        bool_answer = Answer(value=True, answer_type=AnswerType.BOOLEAN)
        assert bool_answer.value is True
        
        # List answer
        list_answer = Answer(value=["A", "B", "C"], answer_type=AnswerType.ENTITY_LIST)
        assert list_answer.value == ["A", "B", "C"]
        
        # Numeric answer
        numeric_answer = Answer(value=3.5, answer_type=AnswerType.NUMERIC)
        assert numeric_answer.value == 3.5
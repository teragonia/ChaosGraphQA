"""Core graph data structures for ChaosGraphQA.

This module defines the fundamental data models for knowledge graphs:
- Entity: Nodes in the graph representing real-world entities
- Relationship: Directed edges connecting entities
- KnowledgeGraph: Container managing entities and their relationships
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

import networkx as nx
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Entity node in the knowledge graph.

    Represents any discrete entity (person, place, object, concept, etc.)
    that can participate in relationships with other entities.
    """

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Human-readable display name")
    entity_type: str = Field(
        default="generic", description="Classification (person, place, concept, etc.)"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata and attributes"
    )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.id == other.id


class Relationship(BaseModel):
    """Directed relationship connecting two entities.

    Represents a typed edge in the graph with optional weight and temporal information.
    Relationships are directional (source → target) and can carry metadata.
    """

    source: str = Field(..., description="Source entity ID (relationship origin)")
    target: str = Field(..., description="Target entity ID (relationship destination)")
    relation_type: str = Field(..., description="Semantic type of relationship")
    weight: Optional[float] = Field(
        default=None, description="Strength or confidence score (0-1)"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Additional relationship metadata"
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="When relationship was established (for temporal reasoning)",
    )

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.relation_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relationship):
            return False
        return (
            self.source == other.source
            and self.target == other.target
            and self.relation_type == other.relation_type
        )


class KnowledgeGraph(BaseModel):
    """Knowledge graph container managing entities and relationships.

    A complete knowledge graph structure with entity dictionary, relationship list,
    and metadata. Provides methods for graph manipulation, querying, and conversion
    to NetworkX for advanced graph algorithms.
    """

    entities: Dict[str, Entity] = Field(
        default_factory=dict, description="Entity dictionary keyed by ID"
    )
    relationships: List[Relationship] = Field(
        default_factory=list, description="List of all relationships between entities"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Graph-level metadata (generator type, complexity, etc.)",
    )

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        self.entities[entity.id] = entity

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph."""
        if relationship.source not in self.entities:
            raise ValueError(f"Source entity {relationship.source} not found in graph")
        if relationship.target not in self.entities:
            raise ValueError(f"Target entity {relationship.target} not found in graph")
        self.relationships.append(relationship)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def get_relationships_from(self, entity_id: str) -> List[Relationship]:
        """Get all relationships originating from an entity."""
        return [r for r in self.relationships if r.source == entity_id]

    def get_relationships_to(self, entity_id: str) -> List[Relationship]:
        """Get all relationships targeting an entity."""
        return [r for r in self.relationships if r.target == entity_id]

    def get_neighbors(
        self, entity_id: str, relation_types: Optional[Set[str]] = None
    ) -> Set[str]:
        """Get all neighboring entities (both incoming and outgoing)."""
        neighbors = set()

        for rel in self.relationships:
            if relation_types and rel.relation_type not in relation_types:
                continue

            if rel.source == entity_id:
                neighbors.add(rel.target)
            elif rel.target == entity_id:
                neighbors.add(rel.source)

        return neighbors

    def to_networkx(self, directed: bool = True) -> Union[nx.DiGraph, nx.Graph]:
        """Convert to NetworkX graph for graph algorithm operations.

        Args:
            directed: If True, creates DiGraph; if False, creates undirected Graph

        Returns:
            NetworkX graph with entity/relationship data as node/edge attributes
        """
        G: Union[nx.DiGraph, nx.Graph] = nx.DiGraph() if directed else nx.Graph()

        for entity in self.entities.values():
            G.add_node(entity.id, **entity.model_dump())

        for rel in self.relationships:
            edge_data = rel.model_dump()
            edge_data.pop("source")
            edge_data.pop("target")
            G.add_edge(rel.source, rel.target, **edge_data)

        return G

    @classmethod
    def from_networkx(cls, G: Union[nx.DiGraph, nx.Graph]) -> "KnowledgeGraph":
        """Create KnowledgeGraph from NetworkX graph."""
        kg = cls()

        # Add entities
        for node_id, node_data in G.nodes(data=True):
            entity_data = node_data.copy()
            if "id" not in entity_data:
                entity_data["id"] = node_id
            if "name" not in entity_data:
                entity_data["name"] = str(node_id)

            entity = Entity(**entity_data)
            kg.add_entity(entity)

        # Add relationships
        for source, target, edge_data in G.edges(data=True):
            rel_data = edge_data.copy()
            rel_data["source"] = source
            rel_data["target"] = target
            if "relation_type" not in rel_data:
                rel_data["relation_type"] = "connected"

            relationship = Relationship(**rel_data)
            kg.add_relationship(relationship)

        return kg

    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the graph."""
        return {
            "num_entities": len(self.entities),
            "num_relationships": len(self.relationships),
            "entity_types": list(set(e.entity_type for e in self.entities.values())),
            "relation_types": list(set(r.relation_type for r in self.relationships)),
            "avg_degree": (
                len(self.relationships) * 2 / len(self.entities) if self.entities else 0
            ),
        }

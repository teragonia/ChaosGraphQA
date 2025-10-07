"""Base generator class for knowledge graphs.

This module provides the abstract base class for all graph generators:
- BaseGraphGenerator: Abstract class with common generation utilities
- Complexity scaling (graph size increases with level 1-4)
- Entity name generation (pronounceable random names)
- Connectivity enforcement
- Random edge addition
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import networkx as nx

from ..models.graph import Entity, KnowledgeGraph, Relationship
from ..models.question import QuestionType


class BaseGraphGenerator(ABC):
    """Abstract base class for all knowledge graph generators.

    Provides common functionality for graph generation including:
    - Automatic complexity scaling
    - Pronounceable entity name generation
    - Connectivity verification
    - Random edge generation
    - Graph validation
    """

    def __init__(
        self,
        complexity_level: int = 1,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize generator with complexity level and random seed.

        Args:
            complexity_level: Difficulty (1=easy, 2=medium, 3=hard, 4=expert)
            seed: Random seed for reproducible generation
            config: Override default complexity parameters
        """
        self.complexity_level = complexity_level
        self.seed = seed
        self.config = config or {}
        self.rng = random.Random(seed)
        self.complexity_params = self._get_complexity_params()

    def _get_complexity_params(self) -> Dict[str, Any]:
        """Calculate graph size parameters based on complexity level.

        Returns:
            Dictionary with min/max nodes and edges for the complexity level
        """
        base_params = {
            1: {"min_nodes": 5, "max_nodes": 10, "min_edges": 8, "max_edges": 15},
            2: {"min_nodes": 15, "max_nodes": 30, "min_edges": 25, "max_edges": 50},
            3: {"min_nodes": 40, "max_nodes": 80, "min_edges": 80, "max_edges": 150},
            4: {"min_nodes": 100, "max_nodes": 200, "min_edges": 200, "max_edges": 400},
        }

        params = base_params.get(self.complexity_level, base_params[1])
        params.update(self.config.get("complexity_params", {}))
        return params

    @abstractmethod
    def generate(self) -> KnowledgeGraph:
        """Generate a knowledge graph.

        Returns:
            Generated knowledge graph
        """
        pass

    @abstractmethod
    def get_question_type(self) -> QuestionType:
        """Get the primary question type this generator supports."""
        pass

    def _generate_entity_names(self, count: int, prefix: str = "E") -> List[str]:
        """Generate unique pronounceable entity names.

        Uses alternating consonant-vowel pattern to create readable names
        that avoid memorization by LLMs.

        Args:
            count: Number of names to generate
            prefix: Prefix letter for entity names (e.g., "N" for nodes)

        Returns:
            List of unique pronounceable entity names
        """
        names = []
        used_names = set()
        consonants = "bcdfghjklmnpqrstvwxz"
        vowels = "aeiou"

        for _ in range(count):
            while True:
                length = self.rng.randint(3, 6)
                name_parts = []

                for j in range(length):
                    if j % 2 == 0:
                        name_parts.append(self.rng.choice(consonants))
                    else:
                        name_parts.append(self.rng.choice(vowels))

                name = f"{prefix}{''.join(name_parts).capitalize()}"

                if name not in used_names:
                    used_names.add(name)
                    names.append(name)
                    break

        return names

    def _create_entities(
        self, count: int, entity_types: Optional[List[str]] = None, prefix: str = "E"
    ) -> Dict[str, Entity]:
        """Create a set of entities."""
        if entity_types is None:
            entity_types = ["person", "place", "concept", "object"]

        names = self._generate_entity_names(count, prefix)
        entities = {}

        for name in names:
            entity_id = f"{name.lower()}_{self.rng.randint(1000, 9999)}"
            entity_type = self.rng.choice(entity_types)

            entity = Entity(
                id=entity_id,
                name=name,
                entity_type=entity_type,
                properties={"generated": True, "complexity": self.complexity_level},
            )

            entities[entity_id] = entity

        return entities

    def _ensure_connectivity(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """Ensure graph connectivity by bridging disconnected components.

        Args:
            kg: Knowledge graph to verify and repair

        Returns:
            Modified knowledge graph with guaranteed connectivity
        """
        nx_graph = kg.to_networkx(directed=False)
        components = list(nx.connected_components(nx_graph))

        if len(components) <= 1:
            return kg

        relation_types = list(set(r.relation_type for r in kg.relationships))
        if not relation_types:
            relation_types = ["connected"]

        for i in range(len(components) - 1):
            node1 = self.rng.choice(list(components[i]))
            node2 = self.rng.choice(list(components[i + 1]))
            rel = Relationship(
                source=node1,
                target=node2,
                relation_type=self.rng.choice(relation_types),
            )
            kg.add_relationship(rel)

        return kg

    def _add_random_edges(
        self,
        kg: KnowledgeGraph,
        num_edges: int,
        relation_types: List[str],
        avoid_self_loops: bool = True,
    ) -> None:
        """Add random edges to the knowledge graph."""
        entity_ids = list(kg.entities.keys())
        existing_edges = {
            (r.source, r.target, r.relation_type) for r in kg.relationships
        }

        attempts = 0
        max_attempts = num_edges * 10  # Avoid infinite loops

        while (
            len(kg.relationships) < len(existing_edges) + num_edges
            and attempts < max_attempts
        ):
            source = self.rng.choice(entity_ids)
            target = self.rng.choice(entity_ids)
            relation_type = self.rng.choice(relation_types)

            if avoid_self_loops and source == target:
                attempts += 1
                continue

            edge_key = (source, target, relation_type)
            if edge_key not in existing_edges:
                rel = Relationship(
                    source=source, target=target, relation_type=relation_type
                )
                kg.add_relationship(rel)
                existing_edges.add(edge_key)

            attempts += 1

    def validate_graph(self, kg: KnowledgeGraph) -> bool:
        """Validate that the generated graph meets requirements."""
        params = self.complexity_params

        # Check size constraints
        if not (params["min_nodes"] <= len(kg.entities) <= params["max_nodes"]):
            return False

        if not (params["min_edges"] <= len(kg.relationships) <= params["max_edges"]):
            return False

        # Check connectivity
        nx_graph = kg.to_networkx(directed=False)
        if not nx.is_connected(nx_graph):
            return False

        return True

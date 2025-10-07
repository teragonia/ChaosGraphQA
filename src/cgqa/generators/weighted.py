"""Weighted reasoning graph generator."""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from ..models.graph import Entity, KnowledgeGraph, Relationship
from ..models.question import QuestionType
from .base_generator import BaseGraphGenerator


class WeightedGenerator(BaseGraphGenerator):
    """Generates graphs optimized for weighted/probabilistic reasoning tasks."""

    def __init__(
        self,
        complexity_level: int = 1,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(complexity_level, seed, config)

        # Weighted specific configuration
        self.weight_ranges = self._get_weight_ranges()
        self.confidence_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.weighted_relation_types = self._get_weighted_relation_types()

        # Ensure sufficient edges for meaningful probability calculations
        min_edges_for_weights = max(
            self.complexity_params["min_nodes"] * 2, self.complexity_params["min_edges"]
        )
        self.complexity_params["min_edges"] = min_edges_for_weights

    def _get_weight_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get different weight ranges for different relationship types."""
        return {
            "high_confidence": (0.8, 1.0),
            "medium_confidence": (0.4, 0.8),
            "low_confidence": (0.1, 0.4),
            "uncertain": (0.0, 0.2),
        }

    def _get_weighted_relation_types(self) -> List[Dict[str, Any]]:
        """Get relationship types that benefit from weights."""
        return [
            {
                "relation": "likely_knows",
                "weight_type": "confidence",
                "description": "Probability that person A knows person B",
            },
            {
                "relation": "probably_located_in",
                "weight_type": "confidence",
                "description": "Probability that entity A is located in B",
            },
            {
                "relation": "may_cause",
                "weight_type": "confidence",
                "description": "Probability that A causes B",
            },
            {
                "relation": "similarity_to",
                "weight_type": "similarity",
                "description": "Similarity score between A and B",
            },
            {
                "relation": "distance_to",
                "weight_type": "distance",
                "description": "Normalized distance between A and B (0=close, 1=far)",
            },
            {
                "relation": "trust_level",
                "weight_type": "trust",
                "description": "Trust score from A to B",
            },
            {
                "relation": "expertise_match",
                "weight_type": "relevance",
                "description": "How well A matches expertise area B",
            },
            {
                "relation": "preference_for",
                "weight_type": "preference",
                "description": "A's preference score for B",
            },
        ]

    def get_question_type(self) -> QuestionType:
        """Return the question type this generator supports."""
        return QuestionType.WEIGHTED

    def generate(self) -> KnowledgeGraph:
        """Generate a knowledge graph optimized for weighted reasoning."""

        # Determine graph size
        num_nodes = self.rng.randint(
            self.complexity_params["min_nodes"], self.complexity_params["max_nodes"]
        )

        # Create entities with different types for meaningful relationships
        entities = self._create_weighted_entities(num_nodes)

        # Create knowledge graph
        kg = KnowledgeGraph()
        for entity in entities.values():
            kg.add_entity(entity)

        # Create weighted relationships
        weighted_relationships = self._create_weighted_relationships(entities)

        for rel in weighted_relationships:
            kg.add_relationship(rel)

        # Ensure connectivity with weighted paths
        kg = self._ensure_weighted_connectivity(kg)

        # Add metadata for question generation
        kg.metadata.update(
            {
                "generator_type": "weighted",
                "complexity_level": self.complexity_level,
                "weight_ranges": self.weight_ranges,
                "confidence_thresholds": self.confidence_thresholds,
                "weighted_paths": self._find_weighted_paths(kg),
                "high_confidence_links": self._find_high_confidence_links(kg),
                "threshold_queries": self._generate_threshold_queries(kg),
            }
        )

        return kg

    def _create_weighted_entities(self, num_entities: int) -> Dict[str, Entity]:
        """Create entities suitable for weighted relationships."""

        entity_types_with_props = [
            {
                "type": "person",
                "properties": {
                    "expertise_level": lambda: self.rng.uniform(0.1, 1.0),
                    "social_score": lambda: self.rng.uniform(0.0, 1.0),
                },
            },
            {
                "type": "location",
                "properties": {
                    "accessibility": lambda: self.rng.uniform(0.2, 1.0),
                    "popularity": lambda: self.rng.uniform(0.0, 1.0),
                },
            },
            {
                "type": "skill",
                "properties": {
                    "difficulty": lambda: self.rng.uniform(0.1, 1.0),
                    "market_demand": lambda: self.rng.uniform(0.0, 1.0),
                },
            },
            {
                "type": "project",
                "properties": {
                    "success_probability": lambda: self.rng.uniform(0.3, 0.9),
                    "resource_requirement": lambda: self.rng.uniform(0.1, 1.0),
                },
            },
        ]

        entities = {}
        entity_names = self._generate_entity_names(num_entities, prefix="W")

        for i, name in enumerate(entity_names):
            entity_config = self.rng.choice(entity_types_with_props)

            # Generate properties
            properties = {"generated": True, "complexity": self.complexity_level}
            for prop_name, prop_generator in entity_config["properties"].items():
                properties[prop_name] = round(prop_generator(), 3)

            entity_id = f"w_{name.lower()}_{self.rng.randint(1000, 9999)}"
            entity = Entity(
                id=entity_id,
                name=name,
                entity_type=entity_config["type"],
                properties=properties,
            )

            entities[entity_id] = entity

        return entities

    def _create_weighted_relationships(
        self, entities: Dict[str, Entity]
    ) -> List[Relationship]:
        """Create weighted relationships between entities."""
        relationships = []
        entity_list = list(entities.values())

        # Calculate target number of relationships
        target_relations = self.rng.randint(
            self.complexity_params["min_edges"], self.complexity_params["max_edges"]
        )

        # Create different types of weighted relationships
        relation_configs = self.weighted_relation_types

        for _ in range(target_relations):
            # Choose two different entities
            if len(entity_list) < 2:
                break

            entity_a, entity_b = self.rng.sample(entity_list, 2)
            relation_config = self.rng.choice(relation_configs)

            # Generate weight based on entity properties and relationship type
            weight = self._calculate_relationship_weight(
                entity_a, entity_b, relation_config
            )

            # Determine weight category
            weight_category = self._classify_weight(weight)

            rel = Relationship(
                source=entity_a.id,
                target=entity_b.id,
                relation_type=relation_config["relation"],
                weight=weight,
                properties={
                    "weight_type": relation_config["weight_type"],
                    "weight_category": weight_category,
                    "description": relation_config["description"],
                },
            )

            relationships.append(rel)

        return relationships

    def _calculate_relationship_weight(
        self, entity_a: Entity, entity_b: Entity, relation_config: Dict[str, Any]
    ) -> float:
        """Calculate relationship weight based on entity properties."""

        relation_type = relation_config["relation"]
        weight_type = relation_config["weight_type"]

        # Base weight with some randomness
        base_weight = self.rng.uniform(0.1, 0.9)

        # Adjust based on entity properties and relationship type
        if (
            relation_type == "likely_knows"
            and entity_a.entity_type == "person"
            and entity_b.entity_type == "person"
        ):
            # Higher weight if both have high social scores
            social_a = entity_a.properties.get("social_score", 0.5)
            social_b = entity_b.properties.get("social_score", 0.5)
            base_weight = (social_a + social_b) / 2 + self.rng.uniform(-0.2, 0.2)

        elif relation_type == "expertise_match" and entity_b.entity_type == "skill":
            # Weight based on person's expertise level
            expertise = entity_a.properties.get("expertise_level", 0.5)
            skill_difficulty = entity_b.properties.get("difficulty", 0.5)
            base_weight = max(
                0.0, expertise - skill_difficulty * 0.5
            ) + self.rng.uniform(-0.1, 0.1)

        elif relation_type == "similarity_to":
            # Random similarity with some structure
            base_weight = self.rng.uniform(0.0, 1.0)
            if entity_a.entity_type == entity_b.entity_type:
                base_weight += 0.2  # Same type entities are more similar

        elif relation_type == "distance_to":
            # Inverse relationship - lower weight means closer
            if (
                entity_a.entity_type == "location"
                and entity_b.entity_type == "location"
            ):
                base_weight = self.rng.uniform(0.0, 1.0)

        # Clamp weight to valid range
        return max(0.0, min(1.0, base_weight))

    def _classify_weight(self, weight: float) -> str:
        """Classify weight into categories."""
        if weight >= 0.8:
            return "high_confidence"
        elif weight >= 0.6:
            return "medium_high_confidence"
        elif weight >= 0.4:
            return "medium_confidence"
        elif weight >= 0.2:
            return "low_confidence"
        else:
            return "very_low_confidence"

    def _ensure_weighted_connectivity(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """Ensure graph connectivity with weighted relationships."""
        nx_graph = kg.to_networkx(directed=False)
        components = list(nx.connected_components(nx_graph))

        if len(components) <= 1:
            return kg

        # Connect components with medium-weight relationships
        for i in range(len(components) - 1):
            node1 = self.rng.choice(list(components[i]))
            node2 = self.rng.choice(list(components[i + 1]))

            rel = Relationship(
                source=node1,
                target=node2,
                relation_type="connected_to",
                weight=self.rng.uniform(0.4, 0.7),  # Medium confidence connection
                properties={
                    "weight_type": "connectivity",
                    "weight_category": "medium_confidence",
                    "description": "Connectivity relationship",
                },
            )
            kg.add_relationship(rel)

        return kg

    def _find_weighted_paths(self, kg: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Find weighted paths for question generation."""
        paths = []

        # Create weighted networkx graph
        G = nx.Graph()
        for rel in kg.relationships:
            if rel.weight is not None:
                # Use (1 - weight) as edge weight so shortest path finds highest confidence path
                G.add_edge(
                    rel.source, rel.target, weight=1 - rel.weight, confidence=rel.weight
                )

        # Find paths between random entity pairs
        entity_ids = list(kg.entities.keys())

        for _ in range(min(20, len(entity_ids) * 2)):  # Limit number of paths
            if len(entity_ids) < 2:
                break

            start, end = self.rng.sample(entity_ids, 2)

            if nx.has_path(G, start, end):
                try:
                    # Find highest confidence path (shortest weighted path)
                    path = nx.shortest_path(G, start, end, weight="weight")

                    if len(path) > 1:
                        # Calculate path confidence (minimum edge confidence)
                        path_confidence = 1.0
                        for i in range(len(path) - 1):
                            edge_data = G.get_edge_data(path[i], path[i + 1])
                            edge_confidence = edge_data.get("confidence", 0.0)
                            path_confidence = min(path_confidence, edge_confidence)

                        paths.append(
                            {
                                "start": start,
                                "end": end,
                                "path": path,
                                "length": len(path) - 1,
                                "confidence": round(path_confidence, 3),
                            }
                        )

                except nx.NetworkXNoPath:
                    continue

        # Sort by confidence and return top paths
        paths.sort(key=lambda x: x["confidence"], reverse=True)
        return paths[:10]

    def _find_high_confidence_links(self, kg: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Find high-confidence direct links."""
        high_confidence_links = []

        for rel in kg.relationships:
            if rel.weight is not None and rel.weight >= 0.8:
                source_entity = kg.get_entity(rel.source)
                target_entity = kg.get_entity(rel.target)

                high_confidence_links.append(
                    {
                        "source": rel.source,
                        "target": rel.target,
                        "source_name": (
                            source_entity.name if source_entity else "Unknown"
                        ),
                        "target_name": (
                            target_entity.name if target_entity else "Unknown"
                        ),
                        "relation_type": rel.relation_type,
                        "confidence": rel.weight,
                    }
                )

        return high_confidence_links

    def _generate_threshold_queries(self, kg: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Generate potential threshold-based queries."""
        queries = []

        for threshold in self.confidence_thresholds:
            # Find relationships above threshold
            above_threshold = [
                rel
                for rel in kg.relationships
                if rel.weight is not None and rel.weight >= threshold
            ]

            # Find entities connected by high-confidence relationships
            high_conf_entities = set()
            for rel in above_threshold:
                high_conf_entities.add(rel.source)
                high_conf_entities.add(rel.target)

            if len(above_threshold) > 0:
                queries.append(
                    {
                        "threshold": threshold,
                        "relationships_above": len(above_threshold),
                        "entities_involved": len(high_conf_entities),
                        "relation_types": list(
                            set(rel.relation_type for rel in above_threshold)
                        ),
                    }
                )

        return queries

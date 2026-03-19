"""Multi-hop reasoning graph generator."""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, cast

import networkx as nx

from ..models.graph import Entity, KnowledgeGraph, Relationship
from ..models.question import QuestionType
from .base_generator import BaseGraphGenerator


class MultiHopGenerator(BaseGraphGenerator):
    """Generates graphs optimized for multi-hop reasoning tasks."""

    def __init__(
        self,
        complexity_level: int = 1,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(complexity_level, seed, config)

        # Multi-hop specific configuration
        self.max_hops = min(2 + complexity_level, 7)  # 3-7 hops max based on complexity
        self.relation_types = self._get_relation_types()

        # Ensure we have enough entities for meaningful paths
        min_nodes_for_hops = max(self.max_hops + 2, self.complexity_params["min_nodes"])
        self.complexity_params["min_nodes"] = min_nodes_for_hops

    def _get_relation_types(self) -> List[str]:
        """Get relation types appropriate for multi-hop reasoning."""
        base_relations = [
            "knows",
            "friend_of",
            "colleague_of",
            "related_to",
            "lives_in",
            "works_at",
            "owns",
            "manages",
            "connected_to",
            "part_of",
            "member_of",
            "leads",
            "teaches",
            "studies_at",
            "born_in",
            "created_by",
        ]

        # More relation types for higher complexity
        num_relations = min(len(base_relations), 4 + self.complexity_level * 2)
        return self.rng.sample(base_relations, num_relations)

    def get_question_type(self) -> QuestionType:
        """Return the question type this generator supports."""
        return QuestionType.MULTIHOP

    def generate(self) -> KnowledgeGraph:
        """Generate a knowledge graph optimized for multi-hop reasoning."""
        # Determine graph size
        num_nodes = self.rng.randint(
            self.complexity_params["min_nodes"], self.complexity_params["max_nodes"]
        )
        num_edges = self.rng.randint(
            self.complexity_params["min_edges"], self.complexity_params["max_edges"]
        )

        # Create entities
        entities = self._create_entities(
            num_nodes,
            entity_types=["person", "organization", "place", "concept"],
            prefix="N",
        )

        # Create knowledge graph
        kg = KnowledgeGraph()
        for entity in entities.values():
            kg.add_entity(entity)

        # Add structured paths for interesting multi-hop questions
        self._add_structured_paths(kg, entities)

        # Fill in remaining edges randomly
        current_edges = len(kg.relationships)
        remaining_edges = num_edges - current_edges

        if remaining_edges > 0:
            self._add_random_edges(kg, remaining_edges, self.relation_types)

        # Ensure connectivity
        kg = self._ensure_connectivity(kg)

        # Add metadata for question generation
        kg.metadata.update(
            {
                "generator_type": "multihop",
                "complexity_level": self.complexity_level,
                "max_hops": self.max_hops,
                "relation_types": self.relation_types,
                "structured_paths": self._find_interesting_paths(kg),
            }
        )

        return kg

    def _add_structured_paths(
        self, kg: KnowledgeGraph, entities: Dict[str, Entity]
    ) -> None:
        """Add structured paths that will create interesting multi-hop questions."""
        entity_ids = list(entities.keys())

        # Create several deliberate paths of different lengths
        paths_to_create = min(3 + self.complexity_level, len(entity_ids) // 3)

        for _ in range(paths_to_create):
            path_length = self.rng.randint(2, self.max_hops)

            # Choose random starting point
            available_nodes = entity_ids.copy()
            path_nodes = [self.rng.choice(available_nodes)]
            available_nodes.remove(path_nodes[0])

            # Build path
            for i in range(path_length):
                if not available_nodes:
                    break

                next_node = self.rng.choice(available_nodes)
                path_nodes.append(next_node)
                available_nodes.remove(next_node)

                # Add relationship
                rel_type = self.rng.choice(self.relation_types)
                rel = Relationship(
                    source=path_nodes[i],
                    target=path_nodes[i + 1],
                    relation_type=rel_type,
                    properties={"is_structured_path": True},
                )
                kg.add_relationship(rel)

        # Add some branching paths to create multiple routes
        self._add_branching_paths(kg, entity_ids)

    def _add_branching_paths(self, kg: KnowledgeGraph, entity_ids: List[str]) -> None:
        """Add branching paths to create alternative routes."""
        existing_sources = {r.source for r in kg.relationships}

        # Add branches from existing path nodes
        for source_id in list(existing_sources)[: self.complexity_level + 1]:
            available_targets = [
                eid
                for eid in entity_ids
                if eid != source_id
                and not any(
                    r.source == source_id and r.target == eid for r in kg.relationships
                )
            ]

            if available_targets:
                # Add 1-2 additional branches
                num_branches = self.rng.randint(1, 2)
                for _ in range(min(num_branches, len(available_targets))):
                    target = self.rng.choice(available_targets)
                    available_targets.remove(target)

                    rel = Relationship(
                        source=source_id,
                        target=target,
                        relation_type=self.rng.choice(self.relation_types),
                        properties={"is_branch_path": True},
                    )
                    kg.add_relationship(rel)

    def _find_interesting_paths(self, kg: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Find paths that would make good multi-hop questions."""
        nx_graph = kg.to_networkx()
        interesting_paths = []

        entity_ids = list(kg.entities.keys())

        # Find paths of various lengths
        for start_id in entity_ids[
            : min(10, len(entity_ids))
        ]:  # Limit to avoid expensive computation
            for end_id in entity_ids:
                if start_id == end_id:
                    continue

                try:
                    # Find shortest paths
                    if nx.has_path(nx_graph, start_id, end_id):
                        shortest_path = nx.shortest_path(nx_graph, start_id, end_id)
                        path_length = len(shortest_path) - 1

                        if 2 <= path_length <= self.max_hops:
                            # Collect relation types along path
                            path_relations = []
                            for i in range(len(shortest_path) - 1):
                                edge_data = nx_graph.get_edge_data(
                                    shortest_path[i], shortest_path[i + 1]
                                )
                                path_relations.append(
                                    edge_data.get("relation_type", "unknown")
                                )

                            interesting_paths.append(
                                {
                                    "start": start_id,
                                    "end": end_id,
                                    "path": shortest_path,
                                    "length": path_length,
                                    "relations": path_relations,
                                }
                            )

                except nx.NetworkXNoPath:
                    continue

        # Sort by path length and diversity
        interesting_paths.sort(
            key=lambda x: (
                cast(int, x["length"]),
                len(set(cast(Iterable[Any], x["relations"]))),
            )
        )

        # Return top paths for question generation
        return interesting_paths[:20]

    def get_path_between(
        self, kg: KnowledgeGraph, start_id: str, end_id: str
    ) -> Optional[List[str]]:
        """Get shortest path between two entities."""
        nx_graph = kg.to_networkx()

        try:
            retval: Optional[List[str]] = nx.shortest_path(nx_graph, start_id, end_id)
            return retval
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_all_paths(
        self,
        kg: KnowledgeGraph,
        start_id: str,
        end_id: str,
        max_length: Optional[int] = None,
    ) -> List[List[str]]:
        """Get all simple paths between two entities."""
        nx_graph = kg.to_networkx()

        if max_length is None:
            max_length = self.max_hops

        try:
            paths = list(
                nx.all_simple_paths(nx_graph, start_id, end_id, cutoff=max_length)
            )
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

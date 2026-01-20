"""Graph algorithms for ground truth verification."""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx

from ..models.graph import KnowledgeGraph
from ..models.relationship_semantics import RelationshipSemantics


class GraphAlgorithms:
    """Collection of graph algorithms for reasoning verification."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.nx_graph = self._build_semantic_directed_graph()
        self.undirected_graph = kg.to_networkx(directed=False)

    def _build_semantic_directed_graph(self) -> nx.DiGraph:
        """Build a directed graph that respects semantic directionality of relationships."""
        graph: nx.DiGraph = nx.DiGraph()

        for entity_id, entity in self.kg.entities.items():
            graph.add_node(entity_id, **entity.properties)

        for rel in self.kg.relationships:
            effective_from, effective_to = (
                RelationshipSemantics.get_effective_direction(
                    rel.source, rel.target, rel.relation_type
                )
            )

            graph.add_edge(
                effective_from,
                effective_to,
                relation_type=rel.relation_type,
                original_source=rel.source,
                original_target=rel.target,
                **rel.properties,
            )

            if RelationshipSemantics.is_bidirectional(rel.relation_type):
                graph.add_edge(
                    effective_to,
                    effective_from,
                    relation_type=rel.relation_type,
                    original_source=rel.source,
                    original_target=rel.target,
                    **rel.properties,
                )

        return graph

    def find_shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find shortest path between two entities."""
        try:
            retval: Optional[List[str]] = nx.shortest_path(self.nx_graph, start, end)
            return retval
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def find_all_shortest_paths(self, start: str, end: str) -> List[List[str]]:
        """Find all shortest paths between two entities."""
        try:
            shortest_length = nx.shortest_path_length(self.nx_graph, start, end)
            all_paths = list(
                nx.all_simple_paths(self.nx_graph, start, end, cutoff=shortest_length)
            )
            return [path for path in all_paths if len(path) - 1 == shortest_length]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def find_all_simple_paths(
        self, start: str, end: str, max_length: Optional[int] = None
    ) -> List[List[str]]:
        """Find all simple paths between two entities."""
        try:
            if max_length is None:
                max_length = len(self.kg.entities)

            paths = list(
                nx.all_simple_paths(self.nx_graph, start, end, cutoff=max_length)
            )
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def path_exists(self, start: str, end: str) -> bool:
        """Check if a path exists between two entities."""
        try:
            return bool(nx.has_path(self.nx_graph, start, end))
        except nx.NodeNotFound:
            return False

    def get_path_length(self, start: str, end: str) -> Optional[int]:
        """Get shortest path length between two entities."""
        try:
            path_length = nx.shortest_path_length(self.nx_graph, start, end)
            return int(path_length)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def find_paths_with_constraints(
        self,
        start: str,
        end: str,
        allowed_relations: Optional[Set[str]] = None,
        forbidden_relations: Optional[Set[str]] = None,
        max_hops: Optional[int] = None,
    ) -> List[List[str]]:
        """Find paths with relationship constraints."""

        if not self.path_exists(start, end):
            return []

        filtered_graph = self._create_filtered_graph(
            allowed_relations, forbidden_relations
        )

        try:
            if max_hops is None:
                max_hops = len(self.kg.entities)

            paths = list(
                nx.all_simple_paths(filtered_graph, start, end, cutoff=max_hops)
            )
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _create_filtered_graph(
        self,
        allowed_relations: Optional[Set[str]] = None,
        forbidden_relations: Optional[Set[str]] = None,
    ) -> nx.DiGraph:
        """Create a filtered graph based on relationship constraints."""
        filtered: nx.DiGraph = nx.DiGraph()
        filtered.add_nodes_from(self.nx_graph.nodes(data=True))

        for u, v, data in self.nx_graph.edges(data=True):
            relation_type = data.get("relation_type", "unknown")

            if allowed_relations and relation_type not in allowed_relations:
                continue
            if forbidden_relations and relation_type in forbidden_relations:
                continue

            filtered.add_edge(u, v, **data)

        return filtered

    def get_neighbors_by_relation(
        self,
        entity: str,
        relation_types: Optional[Set[str]] = None,
        direction: str = "both",  # "in", "out", "both"
    ) -> Dict[str, List[str]]:
        """Get neighbors categorized by relationship type."""
        neighbors = defaultdict(list)

        for rel in self.kg.relationships:
            if relation_types and rel.relation_type not in relation_types:
                continue

            if direction in ["out", "both"] and rel.source == entity:
                neighbors[rel.relation_type].append(rel.target)

            if direction in ["in", "both"] and rel.target == entity:
                neighbors[rel.relation_type].append(rel.source)

        return dict(neighbors)

    def find_connected_components(self) -> List[Set[str]]:
        """Find all connected components in the graph."""
        return [
            set(component)
            for component in nx.connected_components(self.undirected_graph)
        ]

    def is_reachable_within_hops(self, start: str, end: str, max_hops: int) -> bool:
        """Check if target is reachable within specified hops."""
        path_length = self.get_path_length(start, end)
        return path_length is not None and path_length <= max_hops

    def get_entities_within_hops(self, start: str, max_hops: int) -> Set[str]:
        """Get all entities reachable within max_hops."""
        reachable = set()
        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current, hops = queue.popleft()

            if hops < max_hops:
                # Get neighbors
                for neighbor in self.kg.get_neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        reachable.add(neighbor)
                        queue.append((neighbor, hops + 1))

        return reachable

    def find_cycles(self, max_length: int = 10) -> List[List[str]]:
        """Find cycles in the graph."""
        cycles = []

        try:
            if self.nx_graph.is_directed():
                sccs = list(nx.strongly_connected_components(self.nx_graph))
                for scc in sccs:
                    if len(scc) > 1:
                        subgraph = self.nx_graph.subgraph(scc)
                        for node in scc:
                            try:
                                cycle = nx.find_cycle(subgraph, source=node)
                                cycle_nodes = [edge[0] for edge in cycle] + [
                                    cycle[-1][1]
                                ]
                                if len(cycle_nodes) <= max_length:
                                    cycles.append(cycle_nodes)
                            except nx.NetworkXNoCycle:
                                continue
            else:
                cycle_basis = nx.cycle_basis(self.undirected_graph)
                cycles = [cycle for cycle in cycle_basis if len(cycle) <= max_length]
        except Exception:
            pass

        return cycles

    def compute_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """Compute various centrality measures for entities."""
        centralities = {}

        try:
            # Degree centrality
            degree_cent = nx.degree_centrality(self.undirected_graph)

            # Betweenness centrality (sample for large graphs)
            if len(self.kg.entities) > 100:
                betweenness_cent = nx.betweenness_centrality(
                    self.undirected_graph, k=min(100, len(self.kg.entities))
                )
            else:
                betweenness_cent = nx.betweenness_centrality(self.undirected_graph)

            # Closeness centrality
            closeness_cent = nx.closeness_centrality(self.undirected_graph)

            for entity_id in self.kg.entities:
                centralities[entity_id] = {
                    "degree": degree_cent.get(entity_id, 0.0),
                    "betweenness": betweenness_cent.get(entity_id, 0.0),
                    "closeness": closeness_cent.get(entity_id, 0.0),
                }

        except Exception:
            # Return zeros if computation fails
            for entity_id in self.kg.entities:
                centralities[entity_id] = {
                    "degree": 0.0,
                    "betweenness": 0.0,
                    "closeness": 0.0,
                }

        return centralities

    def verify_path_validity(self, path: List[str]) -> bool:
        """Verify that a path is valid in the graph."""
        if len(path) < 2:
            return True  # Single node or empty path is valid

        for i in range(len(path) - 1):
            if not self.nx_graph.has_edge(path[i], path[i + 1]):
                return False

        return True

    def get_path_relations(self, path: List[str]) -> List[str]:
        """Get the relationship types along a path."""
        if len(path) < 2:
            return []

        relations = []
        for i in range(len(path) - 1):
            edge_data = self.nx_graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                relations.append(edge_data.get("relation_type", "unknown"))
            else:
                relations.append("invalid")

        return relations

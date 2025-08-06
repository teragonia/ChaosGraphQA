"""Hierarchical reasoning graph generator."""

import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple
from ..models.graph import KnowledgeGraph, Entity, Relationship
from ..models.question import QuestionType
from .base_generator import BaseGraphGenerator


class HierarchicalGenerator(BaseGraphGenerator):
    """Generates graphs optimized for hierarchical reasoning tasks."""
    
    def __init__(
        self,
        complexity_level: int = 1,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(complexity_level, seed, config)
        
        # Hierarchical specific configuration
        self.max_depth = min(3 + complexity_level, 6)  # 4-6 levels max
        self.branching_factor = [2, 4][min(complexity_level - 1, 1)]  # 2-4 children per node
        self.hierarchy_types = self._get_hierarchy_types()
        
        # Ensure enough entities for meaningful hierarchies
        min_nodes_for_hierarchy = max(self.max_depth * self.branching_factor, 
                                     self.complexity_params["min_nodes"])
        self.complexity_params["min_nodes"] = min_nodes_for_hierarchy
    
    def _get_hierarchy_types(self) -> List[Dict[str, Any]]:
        """Get different types of hierarchical relationships."""
        return [
            {
                "name": "taxonomy", 
                "relation": "is_a",
                "entities": ["animal", "mammal", "carnivore", "feline", "cat", "lion", "tiger"],
                "root_type": "category"
            },
            {
                "name": "organizational",
                "relation": "reports_to", 
                "entities": ["ceo", "director", "manager", "supervisor", "employee"],
                "root_type": "role"
            },
            {
                "name": "geographical",
                "relation": "located_in",
                "entities": ["continent", "country", "state", "city", "district", "street"],
                "root_type": "location"
            },
            {
                "name": "compositional",
                "relation": "part_of",
                "entities": ["system", "subsystem", "component", "module", "unit", "element"],
                "root_type": "structure"
            }
        ]
    
    def get_question_type(self) -> QuestionType:
        """Return the question type this generator supports."""
        return QuestionType.HIERARCHICAL
    
    def generate(self) -> KnowledgeGraph:
        """Generate a knowledge graph optimized for hierarchical reasoning."""
        
        # Choose hierarchy types to implement
        num_hierarchies = min(1 + self.complexity_level // 2, len(self.hierarchy_types))
        selected_hierarchies = self.rng.sample(self.hierarchy_types, num_hierarchies)
        
        # Create knowledge graph
        kg = KnowledgeGraph()
        
        # Generate hierarchies
        hierarchy_metadata = []
        all_roots = []
        
        for i, hierarchy_config in enumerate(selected_hierarchies):
            hierarchy_nodes, hierarchy_edges, root_node = self._create_hierarchy(
                hierarchy_config, prefix=f"H{i}"
            )
            
            # Add entities to graph
            for entity in hierarchy_nodes.values():
                kg.add_entity(entity)
            
            # Add relationships to graph  
            for rel in hierarchy_edges:
                kg.add_relationship(rel)
            
            all_roots.append(root_node)
            hierarchy_metadata.append({
                "type": hierarchy_config["name"],
                "relation": hierarchy_config["relation"],
                "root": root_node,
                "depth": self._calculate_tree_depth(hierarchy_nodes, hierarchy_edges, root_node),
                "nodes": list(hierarchy_nodes.keys())
            })
        
        # Add some cross-hierarchy relationships for complexity
        if len(selected_hierarchies) > 1:
            self._add_cross_hierarchy_links(kg, hierarchy_metadata)
        
        # Fill remaining space with additional entities if needed
        current_size = len(kg.entities)
        target_size = self.complexity_params["min_nodes"]
        
        if current_size < target_size:
            additional_entities = self._create_entities(
                target_size - current_size, 
                entity_types=["concept", "object"],
                prefix="X"
            )
            
            for entity in additional_entities.values():
                kg.add_entity(entity)
            
            # Add some random connections to additional entities
            self._connect_additional_entities(kg, list(additional_entities.keys()), all_roots)
        
        # Add metadata for question generation
        kg.metadata.update({
            "generator_type": "hierarchical",
            "complexity_level": self.complexity_level,
            "max_depth": self.max_depth,
            "branching_factor": self.branching_factor,
            "hierarchies": hierarchy_metadata,
            "inheritance_paths": self._find_inheritance_paths(kg, hierarchy_metadata)
        })
        
        return kg
    
    def _create_hierarchy(
        self, 
        hierarchy_config: Dict[str, Any], 
        prefix: str = "H"
    ) -> Tuple[Dict[str, Entity], List[Relationship], str]:
        """Create a single hierarchy tree."""
        
        relation_type = hierarchy_config["relation"]
        base_entities = hierarchy_config["entities"]
        root_type = hierarchy_config["root_type"]
        
        # Generate entity names based on hierarchy type
        num_entities = self.rng.randint(
            self.max_depth * 2,
            self.max_depth * self.branching_factor
        )
        
        entity_names = []
        if len(base_entities) >= num_entities:
            entity_names = self.rng.sample(base_entities, num_entities)
        else:
            # Use base entities and generate additional ones
            entity_names = base_entities.copy()
            additional_needed = num_entities - len(base_entities)
            generated_names = self._generate_entity_names(additional_needed, prefix)
            entity_names.extend(generated_names)
        
        # Create entities
        entities = {}
        for i, name in enumerate(entity_names):
            entity_id = f"{prefix.lower()}_{name.lower()}_{self.rng.randint(1000, 9999)}"
            entity = Entity(
                id=entity_id,
                name=name.title(),
                entity_type=root_type if i == 0 else "subcategory",
                properties={
                    "hierarchy_level": 0,  # Will be updated when building tree
                    "hierarchy_type": hierarchy_config["name"]
                }
            )
            entities[entity_id] = entity
        
        # Build tree structure
        entity_ids = list(entities.keys())
        root_id = entity_ids[0]
        remaining_ids = entity_ids[1:]
        
        relationships = []
        levels = {root_id: 0}
        
        # BFS tree construction
        queue = [(root_id, 0)]
        
        while queue and remaining_ids:
            parent_id, level = queue.pop(0)
            
            if level >= self.max_depth - 1:
                continue
            
            # Add children to this parent
            num_children = min(
                self.rng.randint(1, self.branching_factor),
                len(remaining_ids)
            )
            
            for _ in range(num_children):
                if not remaining_ids:
                    break
                
                child_id = remaining_ids.pop(0)
                levels[child_id] = level + 1
                
                # Update entity properties
                entities[child_id].properties["hierarchy_level"] = level + 1
                
                # Create relationship (child -> parent for "is_a", parent -> child for "part_of")
                if relation_type in ["is_a", "instance_of"]:
                    rel = Relationship(
                        source=child_id,
                        target=parent_id,
                        relation_type=relation_type
                    )
                else:  # part_of, reports_to, etc.
                    rel = Relationship(
                        source=parent_id,
                        target=child_id,
                        relation_type=relation_type
                    )
                
                relationships.append(rel)
                queue.append((child_id, level + 1))
        
        return entities, relationships, root_id
    
    def _calculate_tree_depth(
        self, 
        entities: Dict[str, Entity], 
        relationships: List[Relationship], 
        root_id: str
    ) -> int:
        """Calculate the maximum depth of the tree."""
        # Build adjacency list
        children = {}
        for rel in relationships:
            parent = rel.target if rel.relation_type in ["is_a", "instance_of"] else rel.source
            child = rel.source if rel.relation_type in ["is_a", "instance_of"] else rel.target
            
            if parent not in children:
                children[parent] = []
            children[parent].append(child)
        
        def get_depth(node_id: str) -> int:
            if node_id not in children:
                return 0
            return 1 + max(get_depth(child) for child in children[node_id])
        
        return get_depth(root_id)
    
    def _add_cross_hierarchy_links(self, kg: KnowledgeGraph, hierarchies: List[Dict]) -> None:
        """Add relationships between different hierarchies."""
        cross_relations = ["related_to", "associated_with", "connected_to"]
        
        num_cross_links = min(self.complexity_level * 2, 5)
        
        for _ in range(num_cross_links):
            # Pick two different hierarchies
            if len(hierarchies) < 2:
                break
            
            h1, h2 = self.rng.sample(hierarchies, 2)
            
            # Pick random nodes from each hierarchy
            node1 = self.rng.choice(h1["nodes"])
            node2 = self.rng.choice(h2["nodes"])
            
            relation = self.rng.choice(cross_relations)
            
            rel = Relationship(
                source=node1,
                target=node2,
                relation_type=relation,
                properties={"cross_hierarchy": True}
            )
            
            try:
                kg.add_relationship(rel)
            except ValueError:
                # Skip if entities don't exist
                continue
    
    def _connect_additional_entities(
        self, 
        kg: KnowledgeGraph, 
        additional_entities: List[str], 
        root_nodes: List[str]
    ) -> None:
        """Connect additional entities to the hierarchy."""
        connect_relations = ["related_to", "associated_with", "member_of"]
        
        for entity_id in additional_entities:
            # Connect to 1-2 hierarchy nodes
            num_connections = self.rng.randint(1, 2)
            
            for _ in range(num_connections):
                target_root = self.rng.choice(root_nodes)
                relation = self.rng.choice(connect_relations)
                
                rel = Relationship(
                    source=entity_id,
                    target=target_root,
                    relation_type=relation
                )
                
                try:
                    kg.add_relationship(rel)
                except ValueError:
                    continue
    
    def _find_inheritance_paths(
        self, 
        kg: KnowledgeGraph, 
        hierarchies: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Find inheritance paths for question generation."""
        paths = []
        
        for hierarchy in hierarchies:
            # Build hierarchy graph
            hierarchy_graph = nx.DiGraph()
            relation_type = hierarchy["relation"]
            
            for rel in kg.relationships:
                if rel.relation_type == relation_type:
                    # Add edge from child to parent for inheritance queries
                    if relation_type in ["is_a", "instance_of"]:
                        hierarchy_graph.add_edge(rel.source, rel.target)
                    else:  # part_of, reports_to, etc.
                        hierarchy_graph.add_edge(rel.target, rel.source)
            
            # Find paths from leaves to root
            # Leaves are nodes with no incoming edges (nothing points to them)
            leaves = [n for n in hierarchy_graph.nodes() if hierarchy_graph.in_degree(n) == 0]
            root_id = hierarchy["root"]
            
            for leaf in leaves:
                if nx.has_path(hierarchy_graph, leaf, root_id):
                    path = nx.shortest_path(hierarchy_graph, leaf, root_id)
                    if len(path) > 1:  # Only include non-trivial paths
                        paths.append({
                            "hierarchy_type": hierarchy["type"],
                            "relation_type": relation_type,
                            "path": path,
                            "length": len(path) - 1,
                            "leaf": leaf,
                            "root": root_id
                        })
        
        return paths
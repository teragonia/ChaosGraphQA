"""Conflicting information graph generator."""

import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from ..models.graph import KnowledgeGraph, Entity, Relationship
from ..models.question import QuestionType
from .base_generator import BaseGraphGenerator


class ConflictingGenerator(BaseGraphGenerator):
    """Generates graphs with conflicting information for consistency reasoning."""
    
    def __init__(
        self,
        complexity_level: int = 1,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(complexity_level, seed, config)
        
        # Conflicting specific configuration
        self.conflict_types = self._get_conflict_types()
        self.base_relations = self._get_base_relations()
        self.conflict_intensity = min(0.1 + complexity_level * 0.1, 0.4)  # 10-40% conflicting info
        
        # Need more relationships to create meaningful conflicts
        min_edges_for_conflicts = max(self.complexity_params["min_nodes"] * 3, 
                                     self.complexity_params["min_edges"])
        self.complexity_params["min_edges"] = min_edges_for_conflicts
    
    def _get_conflict_types(self) -> List[Dict[str, Any]]:
        """Get different types of conflicts that can be created."""
        return [
            {
                "type": "direct_contradiction",
                "description": "A directly contradicts B about the same fact",
                "example": "X is_friend_of Y vs X is_enemy_of Y"
            },
            {
                "type": "transitive_conflict", 
                "description": "Transitive relationships lead to contradiction",
                "example": "A > B > C > A (impossible cycle in ordering)"
            },
            {
                "type": "inheritance_conflict",
                "description": "Conflicting inheritance or classification",
                "example": "X is_a Cat AND X is_a Dog"
            },
            {
                "type": "temporal_conflict",
                "description": "Conflicting temporal relationships", 
                "example": "A before B AND B before A"
            },
            {
                "type": "exclusivity_conflict",
                "description": "Mutually exclusive properties",
                "example": "X is_alive AND X is_dead"
            },
            {
                "type": "capacity_conflict",
                "description": "Impossible capacity/location constraints",
                "example": "Person in multiple exclusive locations simultaneously"
            }
        ]
    
    def _get_base_relations(self) -> Dict[str, List[str]]:
        """Get relationship types that can create conflicts."""
        return {
            "friendship": ["is_friend_of", "is_enemy_of", "is_neutral_to"],
            "hierarchy": ["is_superior_to", "is_subordinate_to", "is_equal_to"],
            "location": ["is_located_in", "is_not_in", "is_outside_of"],
            "ownership": ["owns", "does_not_own", "previously_owned"],
            "temporal": ["is_before", "is_after", "is_simultaneous_with"],
            "state": ["is_active", "is_inactive", "is_suspended"],
            "classification": ["is_a", "is_not_a", "might_be_a"]
        }
    
    def get_question_type(self) -> QuestionType:
        """Return the question type this generator supports."""
        return QuestionType.CONFLICTING
    
    def generate(self) -> KnowledgeGraph:
        """Generate a knowledge graph with conflicting information."""
        
        # Create base entities
        num_nodes = self.rng.randint(
            self.complexity_params["min_nodes"],
            self.complexity_params["max_nodes"]
        )
        
        entities = self._create_entities(
            num_nodes,
            entity_types=["person", "organization", "location", "object", "concept"],
            prefix="C"
        )
        
        # Create knowledge graph
        kg = KnowledgeGraph()
        for entity in entities.values():
            kg.add_entity(entity)
        
        # Create consistent relationships first
        consistent_relationships = self._create_consistent_relationships(entities)
        
        for rel in consistent_relationships:
            kg.add_relationship(rel)
        
        # Introduce conflicts
        conflicts = self._introduce_conflicts(kg)
        
        for conflict_rel in conflicts:
            kg.add_relationship(conflict_rel)
        
        # Ensure connectivity
        kg = self._ensure_connectivity(kg)
        
        # Add metadata for question generation
        kg.metadata.update({
            "generator_type": "conflicting",
            "complexity_level": self.complexity_level,
            "conflict_intensity": self.conflict_intensity,
            "conflict_types": [ct["type"] for ct in self.conflict_types],
            "detected_conflicts": self._detect_conflicts(kg),
            "consistent_subgraphs": self._find_consistent_subgraphs(kg),
            "conflict_resolution_strategies": self._suggest_conflict_resolutions(kg)
        })
        
        return kg
    
    def _create_consistent_relationships(self, entities: Dict[str, Entity]) -> List[Relationship]:
        """Create a base set of consistent relationships."""
        relationships = []
        entity_list = list(entities.values())
        
        # Calculate base number of relationships
        base_relations_count = int(
            self.complexity_params["min_edges"] * (1 - self.conflict_intensity)
        )
        
        for _ in range(base_relations_count):
            if len(entity_list) < 2:
                break
            
            entity_a, entity_b = self.rng.sample(entity_list, 2)
            
            # Choose relationship based on entity types
            relation_type = self._choose_appropriate_relation(entity_a, entity_b)
            
            rel = Relationship(
                source=entity_a.id,
                target=entity_b.id,
                relation_type=relation_type,
                properties={
                    "consistent": True,
                    "conflict_source": False
                }
            )
            
            relationships.append(rel)
        
        return relationships
    
    def _choose_appropriate_relation(self, entity_a: Entity, entity_b: Entity) -> str:
        """Choose an appropriate relationship type based on entity types."""
        
        if entity_a.entity_type == "person" and entity_b.entity_type == "person":
            return self.rng.choice(self.base_relations["friendship"])
        elif entity_a.entity_type == "person" and entity_b.entity_type == "location":
            return self.rng.choice(self.base_relations["location"])
        elif entity_a.entity_type == "person" and entity_b.entity_type == "object":
            return self.rng.choice(self.base_relations["ownership"])
        elif entity_a.entity_type == "organization" and entity_b.entity_type == "person":
            return self.rng.choice(self.base_relations["hierarchy"])
        else:
            # Generic relationships for other combinations
            return self.rng.choice([
                "related_to", "connected_to", "associated_with", 
                "interacts_with", "depends_on"
            ])
    
    def _introduce_conflicts(self, kg: KnowledgeGraph) -> List[Relationship]:
        """Introduce conflicting relationships to the graph."""
        conflict_relationships = []
        
        # Calculate number of conflicts to introduce
        target_conflicts = int(len(kg.relationships) * self.conflict_intensity)
        
        # Generate different types of conflicts
        conflict_methods = [
            self._create_direct_contradictions,
            self._create_transitive_conflicts,
            self._create_inheritance_conflicts,
            self._create_temporal_conflicts,
            self._create_exclusivity_conflicts
        ]
        
        conflicts_per_type = max(1, target_conflicts // len(conflict_methods))
        
        for method in conflict_methods:
            conflicts = method(kg, conflicts_per_type)
            conflict_relationships.extend(conflicts)
        
        return conflict_relationships[:target_conflicts]  # Limit to target
    
    def _create_direct_contradictions(self, kg: KnowledgeGraph, count: int) -> List[Relationship]:
        """Create direct contradictory relationships."""
        contradictions = []
        
        # Find existing relationships that can be contradicted
        existing_relations = list(kg.relationships)
        
        for _ in range(min(count, len(existing_relations))):
            if not existing_relations:
                break
            
            base_rel = self.rng.choice(existing_relations)
            existing_relations.remove(base_rel)
            
            # Create opposite relationship
            opposite_relation = self._get_opposite_relation(base_rel.relation_type)
            
            if opposite_relation:
                conflict_rel = Relationship(
                    source=base_rel.source,
                    target=base_rel.target,
                    relation_type=opposite_relation,
                    properties={
                        "consistent": False,
                        "conflict_source": True,
                        "conflict_type": "direct_contradiction",
                        "contradicts": base_rel.relation_type
                    }
                )
                contradictions.append(conflict_rel)
        
        return contradictions
    
    def _get_opposite_relation(self, relation: str) -> Optional[str]:
        """Get the opposite/contradictory relation."""
        opposites = {
            "is_friend_of": "is_enemy_of",
            "is_enemy_of": "is_friend_of",
            "is_superior_to": "is_subordinate_to",
            "is_subordinate_to": "is_superior_to",
            "is_located_in": "is_not_in",
            "is_not_in": "is_located_in",
            "owns": "does_not_own",
            "does_not_own": "owns",
            "is_before": "is_after",
            "is_after": "is_before",
            "is_active": "is_inactive",
            "is_inactive": "is_active",
            "is_alive": "is_dead",
            "is_dead": "is_alive"
        }
        
        return opposites.get(relation)
    
    def _create_transitive_conflicts(self, kg: KnowledgeGraph, count: int) -> List[Relationship]:
        """Create conflicts through transitive relationships."""
        conflicts = []
        
        # Look for chains that could form cycles
        transitive_relations = ["is_superior_to", "is_before", "is_greater_than", "depends_on"]
        
        for _ in range(count):
            # Find entities that could form a conflicting cycle
            entity_ids = list(kg.entities.keys())
            if len(entity_ids) < 3:
                break
            
            # Create a 3-cycle: A -> B -> C -> A
            a, b, c = self.rng.sample(entity_ids, 3)
            relation = self.rng.choice(transitive_relations)
            
            # The third relationship creates the conflict
            conflict_rel = Relationship(
                source=c,
                target=a,
                relation_type=relation,
                properties={
                    "consistent": False,
                    "conflict_source": True,
                    "conflict_type": "transitive_conflict",
                    "creates_cycle": True
                }
            )
            conflicts.append(conflict_rel)
        
        return conflicts
    
    def _create_inheritance_conflicts(self, kg: KnowledgeGraph, count: int) -> List[Relationship]:
        """Create conflicting inheritance/classification relationships."""
        conflicts = []
        
        # Create mutually exclusive categories
        exclusive_categories = [
            ["animal", "plant", "mineral"],
            ["living", "non_living"],
            ["human", "animal", "object"],
            ["solid", "liquid", "gas"]
        ]
        
        for _ in range(count):
            entity_ids = list(kg.entities.keys())
            if not entity_ids:
                break
            
            entity_id = self.rng.choice(entity_ids)
            categories = self.rng.choice(exclusive_categories)
            
            if len(categories) >= 2:
                # Create conflicting classifications by creating two categories
                cat1, cat2 = self.rng.sample(categories, 2)
                
                # Create the category entities if they don't exist
                cat1_id = f"category_{cat1}"
                cat2_id = f"category_{cat2}"
                
                if cat1_id not in kg.entities:
                    cat1_entity = Entity(
                        id=cat1_id,
                        name=cat1.title(),
                        entity_type="category",
                        properties={"artificial": True}
                    )
                    kg.add_entity(cat1_entity)
                
                if cat2_id not in kg.entities:
                    cat2_entity = Entity(
                        id=cat2_id,
                        name=cat2.title(),
                        entity_type="category", 
                        properties={"artificial": True}
                    )
                    kg.add_entity(cat2_entity)
                
                # Create conflicting relationships
                conflict_rel1 = Relationship(
                    source=entity_id,
                    target=cat1_id,
                    relation_type="is_a",
                    properties={
                        "consistent": True,
                        "conflict_source": False
                    }
                )
                
                conflict_rel2 = Relationship(
                    source=entity_id,
                    target=cat2_id,
                    relation_type="is_a",
                    properties={
                        "consistent": False,
                        "conflict_source": True,
                        "conflict_type": "inheritance_conflict",
                        "exclusive_with": cat1
                    }
                )
                conflicts.extend([conflict_rel1, conflict_rel2])
        
        return conflicts
    
    def _create_temporal_conflicts(self, kg: KnowledgeGraph, count: int) -> List[Relationship]:
        """Create temporal conflicts (A before B AND B before A)."""
        conflicts = []
        
        for _ in range(count):
            entity_ids = list(kg.entities.keys())
            if len(entity_ids) < 2:
                break
            
            a, b = self.rng.sample(entity_ids, 2)
            
            # Create temporal conflict: if A is_before B, then B cannot be before A
            conflict_rel = Relationship(
                source=b,
                target=a,
                relation_type="is_before",
                properties={
                    "consistent": False,
                    "conflict_source": True,
                    "conflict_type": "temporal_conflict",
                    "conflicts_with": f"{a} is_before {b}"
                }
            )
            conflicts.append(conflict_rel)
        
        return conflicts
    
    def _create_exclusivity_conflicts(self, kg: KnowledgeGraph, count: int) -> List[Relationship]:
        """Create exclusivity conflicts (mutually exclusive states)."""
        conflicts = []
        
        exclusive_states = [
            ["is_open", "is_closed"],
            ["is_alive", "is_dead"],
            ["is_present", "is_absent"],
            ["is_full", "is_empty"]
        ]
        
        for _ in range(count):
            entity_ids = list(kg.entities.keys())
            if not entity_ids:
                break
            
            entity_id = self.rng.choice(entity_ids)
            states = self.rng.choice(exclusive_states)
            
            # Create both states for the same entity (conflict)
            for state in states:
                state_id = f"state_{state}"
                
                # Create the state entity if it doesn't exist
                if state_id not in kg.entities:
                    state_entity = Entity(
                        id=state_id,
                        name=state.replace("_", " ").title(),
                        entity_type="state",
                        properties={"artificial": True}
                    )
                    kg.add_entity(state_entity)
                
                conflict_rel = Relationship(
                    source=entity_id,
                    target=state_id,
                    relation_type="has_state",
                    properties={
                        "consistent": False,
                        "conflict_source": True,
                        "conflict_type": "exclusivity_conflict",
                        "state": state,
                        "mutually_exclusive": True
                    }
                )
                conflicts.append(conflict_rel)
        
        return conflicts[:count]  # Limit to requested count
    
    def _detect_conflicts(self, kg: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Detect and catalog conflicts in the graph."""
        conflicts = []
        
        # Group relationships by source-target pair
        rel_groups = {}
        for rel in kg.relationships:
            key = (rel.source, rel.target)
            if key not in rel_groups:
                rel_groups[key] = []
            rel_groups[key].append(rel)
        
        # Find direct contradictions
        for (source, target), relations in rel_groups.items():
            if len(relations) > 1:
                relation_types = [r.relation_type for r in relations]
                
                # Check for contradictory relations
                for rel_type in relation_types:
                    opposite = self._get_opposite_relation(rel_type)
                    if opposite and opposite in relation_types:
                        conflicts.append({
                            "type": "direct_contradiction",
                            "source": source,
                            "target": target,
                            "conflicting_relations": [rel_type, opposite],
                            "severity": "high"
                        })
        
        # Find transitive conflicts (cycles in ordering relations)
        conflicts.extend(self._find_transitive_conflicts(kg))
        
        return conflicts
    
    def _find_transitive_conflicts(self, kg: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Find transitive conflicts (cycles in ordering relations)."""
        conflicts = []
        
        transitive_relations = ["is_superior_to", "is_before", "is_greater_than", "depends_on"]
        
        for relation_type in transitive_relations:
            # Build directed graph for this relation type
            G = nx.DiGraph()
            for rel in kg.relationships:
                if rel.relation_type == relation_type:
                    G.add_edge(rel.source, rel.target)
            
            # Find cycles
            try:
                cycles = list(nx.simple_cycles(G))
                for cycle in cycles:
                    conflicts.append({
                        "type": "transitive_conflict",
                        "cycle": cycle,
                        "relation_type": relation_type,
                        "severity": "medium"
                    })
            except:
                # Handle large graphs where cycle detection might be expensive
                pass
        
        return conflicts
    
    def _find_consistent_subgraphs(self, kg: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Find maximal consistent subgraphs."""
        subgraphs = []
        
        # Identify conflicting relationships
        conflicting_rels = [
            rel for rel in kg.relationships 
            if not rel.properties.get("consistent", True)
        ]
        
        # Remove conflicting relationships and find connected components
        consistent_rels = [
            rel for rel in kg.relationships 
            if rel.properties.get("consistent", True)
        ]
        
        # Build graph from consistent relationships
        G = nx.Graph()
        for rel in consistent_rels:
            G.add_edge(rel.source, rel.target)
        
        # Find connected components
        components = list(nx.connected_components(G))
        
        for i, component in enumerate(components):
            if len(component) > 1:  # Only include non-trivial components
                subgraphs.append({
                    "id": f"consistent_subgraph_{i}",
                    "entities": list(component),
                    "size": len(component),
                    "relationships": len([
                        rel for rel in consistent_rels
                        if rel.source in component and rel.target in component
                    ])
                })
        
        return subgraphs
    
    def _suggest_conflict_resolutions(self, kg: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Suggest strategies for resolving conflicts."""
        strategies = []
        
        conflicts = kg.metadata.get("detected_conflicts", [])
        
        for conflict in conflicts:
            if conflict["type"] == "direct_contradiction":
                strategies.append({
                    "conflict_id": f"{conflict['source']}_{conflict['target']}",
                    "strategy": "choose_most_reliable_source",
                    "description": "Select the relationship from the most trustworthy source",
                    "alternatives": [
                        "temporal_ordering: Keep the more recent information",
                        "source_priority: Trust higher-authority sources",
                        "majority_consensus: Go with the most commonly stated relationship"
                    ]
                })
            
            elif conflict["type"] == "transitive_conflict":
                strategies.append({
                    "conflict_id": str(conflict.get("cycle", [])),
                    "strategy": "break_weakest_link",
                    "description": "Remove the least certain relationship in the cycle",
                    "alternatives": [
                        "reorder_priorities: Change the ordering criterion",
                        "add_context: Make relationships context-dependent"
                    ]
                })
        
        return strategies
"""Question template system for generating reasoning questions."""

import random
from typing import Dict, List, Optional, Any, Tuple
from ..models.graph import KnowledgeGraph
from ..models.question import (
    Question, QuestionType, Answer, AnswerType, 
    QuestionTemplate, QuestionSet
)


class QuestionGenerator:
    """Main question generation system."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.template_registry: Dict[QuestionType, List[QuestionTemplate]] = {}
        self._register_default_templates()
    
    def _register_default_templates(self) -> None:
        """Register default question templates."""
        # Multi-hop templates
        multihop_templates = MultiHopTemplates()
        self.template_registry[QuestionType.MULTIHOP] = multihop_templates.get_templates()
        
        # Hierarchical templates
        hierarchical_templates = HierarchicalTemplates()
        self.template_registry[QuestionType.HIERARCHICAL] = hierarchical_templates.get_templates()
        
        # Temporal templates
        temporal_templates = TemporalTemplates()
        self.template_registry[QuestionType.TEMPORAL] = temporal_templates.get_templates()
        
        # Weighted templates  
        weighted_templates = WeightedTemplates()
        self.template_registry[QuestionType.WEIGHTED] = weighted_templates.get_templates()
        
        # Conflicting templates
        conflicting_templates = ConflictingTemplates() 
        self.template_registry[QuestionType.CONFLICTING] = conflicting_templates.get_templates()
    
    def register_template(self, template: QuestionTemplate) -> None:
        """Register a new question template."""
        if template.question_type not in self.template_registry:
            self.template_registry[template.question_type] = []
        self.template_registry[template.question_type].append(template)
    
    def generate_questions(
        self,
        kg: KnowledgeGraph,
        question_types: Optional[List[QuestionType]] = None,
        num_questions_per_type: int = 5,
        complexity_levels: Optional[List[int]] = None
    ) -> QuestionSet:
        """Generate a set of questions for a knowledge graph."""
        
        if question_types is None:
            question_types = list(self.template_registry.keys())
        
        if complexity_levels is None:
            complexity_levels = [1, 2, 3, 4]
        
        question_set = QuestionSet(
            id=f"qs_{self.rng.randint(10000, 99999)}",
            name="Generated Question Set",
            metadata={
                "graph_stats": kg.get_stats(),
                "generation_params": {
                    "question_types": question_types,
                    "num_questions_per_type": num_questions_per_type,
                    "complexity_levels": complexity_levels
                }
            }
        )
        
        for question_type in question_types:
            if question_type not in self.template_registry:
                continue
            
            templates = self.template_registry[question_type]
            
            for complexity in complexity_levels:
                # Filter templates by complexity
                suitable_templates = [
                    t for t in templates 
                    if t.complexity_level <= complexity and t.can_generate(kg.metadata)
                ]
                
                if not suitable_templates:
                    continue
                
                # Generate questions for this complexity level
                questions_generated = 0
                attempts = 0
                max_attempts = num_questions_per_type * 10
                
                while questions_generated < num_questions_per_type and attempts < max_attempts:
                    template = self.rng.choice(suitable_templates)
                    
                    try:
                        question = self._generate_question_from_template(kg, template, complexity)
                        if question:
                            question_set.add_question(question)
                            questions_generated += 1
                    except Exception as e:
                        # Log error but continue
                        print(f"Failed to generate question: {e}")
                    
                    attempts += 1
        
        return question_set
    
    def _generate_question_from_template(
        self,
        kg: KnowledgeGraph,
        template: QuestionTemplate,
        complexity: int
    ) -> Optional[Question]:
        """Generate a specific question from a template."""
        
        if template.question_type == QuestionType.MULTIHOP:
            return self._generate_multihop_question(kg, template, complexity)
        elif template.question_type == QuestionType.HIERARCHICAL:
            return self._generate_hierarchical_question(kg, template, complexity)
        elif template.question_type == QuestionType.TEMPORAL:
            return self._generate_temporal_question(kg, template, complexity)
        elif template.question_type == QuestionType.WEIGHTED:
            return self._generate_weighted_question(kg, template, complexity)
        elif template.question_type == QuestionType.CONFLICTING:
            return self._generate_conflicting_question(kg, template, complexity)
        
        return None
    
    def _generate_multihop_question(
        self,
        kg: KnowledgeGraph,
        template: QuestionTemplate,
        complexity: int
    ) -> Optional[Question]:
        """Generate a multi-hop reasoning question."""
        
        # Get interesting paths from metadata
        structured_paths = kg.metadata.get("structured_paths", [])
        
        if not structured_paths:
            return None
        
        # Filter paths by complexity (path length)
        suitable_paths = [
            path for path in structured_paths 
            if path["length"] >= template.complexity_level
        ]
        
        if not suitable_paths:
            return None
        
        # Choose a random path
        chosen_path = self.rng.choice(suitable_paths)
        start_entity = kg.get_entity(chosen_path["start"])
        end_entity = kg.get_entity(chosen_path["end"])
        
        if not start_entity or not end_entity:
            return None
        
        # Fill template variables
        variables = {
            "start_entity": start_entity.name,
            "end_entity": end_entity.name,
            "num_hops": str(chosen_path["length"]),
            "relation_types": ", ".join(set(chosen_path["relations"]))
        }
        
        # Generate question text
        try:
            question_text = template.format_question(variables)
        except ValueError:
            return None
        
        # Create ground truth answer - always use entity names for consistency
        if template.answer_type == AnswerType.PATH:
            path_names = []
            for eid in chosen_path["path"]:
                entity = kg.get_entity(eid)
                if entity:
                    path_names.append(entity.name)
            answer_value = " → ".join(path_names)
            explanation = f"The path from {start_entity.name} to {end_entity.name} is: {answer_value}"
        elif template.answer_type == AnswerType.BOOLEAN:
            answer_value = True  # Path exists
            explanation = f"Yes, there is a path from {start_entity.name} to {end_entity.name} in {chosen_path['length']} hops"
        elif template.answer_type == AnswerType.ENTITY_LIST:
            # Return intermediate entity names (not IDs)
            intermediate_entities = []
            for eid in chosen_path["path"][1:-1]:
                entity = kg.get_entity(eid)
                if entity:
                    intermediate_entities.append(entity.name)
            answer_value = intermediate_entities
            explanation = f"Intermediate entities: {', '.join(intermediate_entities) if intermediate_entities else 'none'}"
        elif template.answer_type == AnswerType.NUMERIC:
            # CRITICAL FIX: For step counting questions, verify actual shortest path length
            # Don't just use metadata path, which might not be the shortest
            if "steps does it take" in template.template.lower() or "how many steps" in template.template.lower():
                # This is a step counting question - find the actual shortest path
                actual_shortest_length = self._find_shortest_path_length(kg, chosen_path["start"], chosen_path["end"])
                if actual_shortest_length is not None:
                    answer_value = actual_shortest_length
                    explanation = f"It takes {answer_value} steps to get from {start_entity.name} to {end_entity.name}"
                else:
                    # No path exists
                    return None
            else:
                # Other numeric questions can use the chosen path length
                answer_value = chosen_path["length"]  # Number of hops
                explanation = f"The path length is {answer_value}"
        else:
            answer_value = end_entity.name  # Always use entity name, not ID
            explanation = f"Following the path leads to {end_entity.name}"
        
        ground_truth = Answer(
            value=answer_value,
            answer_type=template.answer_type,
            explanation=explanation,
            metadata={"ground_truth_path": chosen_path}
        )
        
        # Create question
        question = Question(
            id=f"q_{self.rng.randint(100000, 999999)}",
            question_text=question_text,
            question_type=template.question_type,
            complexity_level=complexity,
            ground_truth=ground_truth,
            context_entities=[chosen_path["start"], chosen_path["end"]],
            metadata={
                "template_id": f"{template.question_type}_{template.complexity_level}",
                "path_info": chosen_path
            }
        )
        
        return question
    
    def _generate_hierarchical_question(
        self,
        kg: KnowledgeGraph,
        template: QuestionTemplate,
        complexity: int
    ) -> Optional[Question]:
        """Generate a hierarchical reasoning question."""
        
        inheritance_paths = kg.metadata.get("inheritance_paths", [])
        
        if not inheritance_paths:
            return None
        
        # Choose a random inheritance path
        chosen_path = self.rng.choice(inheritance_paths)
        
        leaf_entity = kg.get_entity(chosen_path["leaf"])
        root_entity = kg.get_entity(chosen_path["root"])
        
        if not leaf_entity or not root_entity:
            return None
        
        # CRITICAL FIX: Verify the path actually exists in the knowledge graph relationships
        # before generating a question about it
        actual_path_length = self._verify_hierarchical_path(kg, chosen_path)
        if actual_path_length is None:
            # If path doesn't exist in actual relationships, return None to skip this question
            return None
        
        # Fill template variables
        variables = {
            "leaf_entity": leaf_entity.name,
            "root_entity": root_entity.name,
            "hierarchy_type": chosen_path["hierarchy_type"],
            "relation_type": chosen_path["relation_type"],
            "path_length": str(actual_path_length)  # Use verified path length
        }
        
        # Generate question text
        try:
            question_text = template.format_question(variables)
        except ValueError:
            return None
        
        # Create ground truth answer using VERIFIED path information
        if template.answer_type == AnswerType.BOOLEAN:
            answer_value = actual_path_length > 0  # Path exists and has length > 0
            if answer_value:
                explanation = f"Yes, {leaf_entity.name} connects to {root_entity.name} through {chosen_path['relation_type']} relationships"
            else:
                explanation = f"No, {leaf_entity.name} does not connect to {root_entity.name} through {chosen_path['relation_type']} relationships"
        elif template.answer_type == AnswerType.PATH:
            # Build actual path from relationships
            actual_path_entities = self._get_actual_hierarchical_path(kg, chosen_path)
            if actual_path_entities:
                path_names = [kg.get_entity(eid).name for eid in actual_path_entities if kg.get_entity(eid)]
                answer_value = " → ".join(path_names)
                explanation = f"The path is: {answer_value}"
            else:
                return None  # Can't generate valid path question
        elif template.answer_type == AnswerType.NUMERIC:
            answer_value = actual_path_length  # Use verified length
            if actual_path_length > 0:
                explanation = f"There are {answer_value} levels between {leaf_entity.name} and {root_entity.name}"
            else:
                explanation = f"There are no {chosen_path['relation_type']} relationships connecting {leaf_entity.name} and {root_entity.name}"
        else:
            if actual_path_length > 0:
                answer_value = root_entity.name  # Always use entity name, not ID
                explanation = f"The root entity is {root_entity.name}"
            else:
                return None  # Can't generate valid question if no path exists
        
        ground_truth = Answer(
            value=answer_value,
            answer_type=template.answer_type,
            explanation=explanation,
            metadata={"inheritance_path": chosen_path, "verified_length": actual_path_length}
        )
        
        return Question(
            id=f"q_{self.rng.randint(100000, 999999)}",
            question_text=question_text,
            question_type=template.question_type,
            complexity_level=complexity,
            ground_truth=ground_truth,
            context_entities=[chosen_path["leaf"], chosen_path["root"]],
            metadata={"template_id": f"{template.question_type}_{template.complexity_level}"}
        )
    
    def _generate_temporal_question(
        self,
        kg: KnowledgeGraph,
        template: QuestionTemplate,
        complexity: int
    ) -> Optional[Question]:
        """Generate a temporal reasoning question."""
        
        temporal_sequences = kg.metadata.get("temporal_sequences", [])
        causal_chains = kg.metadata.get("causal_chains", [])
        events = kg.metadata.get("events", [])
        
        # Always use entity names consistently for all temporal questions
        if template.variables.get("sequence_type") == "causal" and causal_chains:
            chosen_data = self.rng.choice(causal_chains)
            
            # Convert entity IDs to names for consistency
            cause_entity = kg.get_entity(chosen_data["cause"])
            effect_entity = kg.get_entity(chosen_data["effect"])
            
            if not cause_entity or not effect_entity:
                return None
                
            variables = {
                "cause_event": cause_entity.name,
                "effect_event": effect_entity.name,
                "chain_length": str(chosen_data["length"])
            }
            
            # Convert all entities in chain to names
            chain_names = []
            for entity_id in chosen_data["chain"]:
                entity = kg.get_entity(entity_id)
                if entity:
                    chain_names.append(entity.name)
            
            if template.answer_type == AnswerType.SINGLE_ENTITY:
                answer_value = effect_entity.name  # Always use name, not ID
            else:
                answer_value = chain_names
                
            explanation = f"Causal chain: {' → '.join(chain_names)}"
            context_entity_ids = chosen_data["chain"][:2]  # Still use IDs for context
            
        elif temporal_sequences:
            chosen_sequence = self.rng.choice(temporal_sequences)
            event_names = chosen_sequence["event_names"]
            event_ids = chosen_sequence["events"]
            
            variables = {
                "first_event": event_names[0],
                "last_event": event_names[-1],
                "sequence_length": str(chosen_sequence["length"]),
                "start_time": chosen_sequence["start_time"],
                "end_time": chosen_sequence["end_time"]
            }
            
            if template.answer_type == AnswerType.ENTITY_LIST:
                answer_value = event_names[1:-1]  # Intermediate events (names)
                explanation = f"Events between {event_names[0]} and {event_names[-1]}: {', '.join(answer_value) if answer_value else 'none'}"
            else:
                # For "last event" questions, find the longest sequence starting with the first event
                if "last event" in template.template.lower():
                    first_event_name = event_names[0]
                    # Find all sequences starting with this event
                    sequences_starting_with_first = [
                        seq for seq in temporal_sequences 
                        if seq["event_names"] and seq["event_names"][0] == first_event_name
                    ]
                    
                    # Use the longest sequence for ground truth
                    if sequences_starting_with_first:
                        longest_sequence = max(sequences_starting_with_first, key=lambda s: s["length"])
                        answer_value = longest_sequence["event_names"][-1]  # Last event in longest sequence
                        explanation = f"The last event in the longest sequence starting with {first_event_name} is {answer_value}"
                    else:
                        answer_value = event_names[-1]
                        explanation = f"The last event in the sequence is {event_names[-1]}"
                else:
                    answer_value = event_names[-1]  # Last event (name)
                    explanation = f"The last event in the sequence is {event_names[-1]}"
            
            context_entity_ids = event_ids  # Use entity IDs for context
            
        else:
            return None
        
        try:
            question_text = template.format_question(variables)
        except (ValueError, KeyError):
            return None
        
        ground_truth = Answer(
            value=answer_value,
            answer_type=template.answer_type,
            explanation=explanation
        )
        
        return Question(
            id=f"q_{self.rng.randint(100000, 999999)}",
            question_text=question_text,
            question_type=template.question_type,
            complexity_level=complexity,
            ground_truth=ground_truth,
            context_entities=context_entity_ids,
            metadata={"template_id": f"{template.question_type}_{template.complexity_level}"}
        )
    
    def _check_entities_connected(self, kg: KnowledgeGraph, entity_ids: List[str]) -> bool:
        """Check if a set of entities are connected through consistent relationships."""
        import networkx as nx
        
        # Build a graph of only consistent relationships
        G = nx.Graph()
        for rel in kg.relationships:
            if rel.properties.get("consistent", True):
                # Only include relationships between the entities we're checking
                if rel.source in entity_ids and rel.target in entity_ids:
                    G.add_edge(rel.source, rel.target)
        
        # Check if all entities are in the same connected component
        if len(entity_ids) <= 1:
            return True
        
        # Add all entity nodes even if they have no edges
        for entity_id in entity_ids:
            G.add_node(entity_id)
        
        # Find connected components
        components = list(nx.connected_components(G))
        
        # Check if all entities are in the same component
        entity_set = set(entity_ids)
        for component in components:
            if entity_set.issubset(component):
                return True
        
        return False
    
    def _verify_hierarchical_path(self, kg: KnowledgeGraph, path_info: Dict[str, Any]) -> Optional[int]:
        """Verify that a hierarchical path actually exists in the knowledge graph relationships."""
        import networkx as nx
        
        relation_type = path_info["relation_type"]
        start_entity = path_info["leaf"]
        end_entity = path_info["root"]
        
        # Build directed graph from actual relationships
        G = nx.DiGraph()
        for rel in kg.relationships:
            if rel.relation_type == relation_type:
                # For hierarchical relationships, follow the actual direction
                # part_of: A part_of B means A is part of B (A -> B in hierarchy)
                # is_a: A is_a B means A is a type of B (A -> B in hierarchy)
                G.add_edge(rel.source, rel.target)
        
        # Check if path exists
        if nx.has_path(G, start_entity, end_entity):
            try:
                path = nx.shortest_path(G, start_entity, end_entity)
                return len(path) - 1  # Number of relationships (edges)
            except nx.NetworkXNoPath:
                return 0
        else:
            return 0
    
    def _get_actual_hierarchical_path(self, kg: KnowledgeGraph, path_info: Dict[str, Any]) -> Optional[List[str]]:
        """Get the actual hierarchical path from relationships."""
        import networkx as nx
        
        relation_type = path_info["relation_type"]
        start_entity = path_info["leaf"]
        end_entity = path_info["root"]
        
        # Build directed graph from actual relationships
        G = nx.DiGraph()
        for rel in kg.relationships:
            if rel.relation_type == relation_type:
                G.add_edge(rel.source, rel.target)
        
        # Get actual path
        if nx.has_path(G, start_entity, end_entity):
            try:
                return nx.shortest_path(G, start_entity, end_entity)
            except nx.NetworkXNoPath:
                return None
        else:
            return None
    
    def _verify_weighted_path(self, kg: KnowledgeGraph, path_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Verify that a weighted path actually exists in the knowledge graph relationships."""
        import networkx as nx
        
        start_entity = path_info["start"]
        end_entity = path_info["end"]
        
        # Build weighted graph from actual relationships
        G = nx.Graph()  # Use undirected graph for weighted paths
        for rel in kg.relationships:
            if rel.weight is not None:
                # Use (1 - weight) as edge weight so shortest path finds highest confidence path
                G.add_edge(rel.source, rel.target, weight=1 - rel.weight, confidence=rel.weight)
        
        # Check if path exists
        if nx.has_path(G, start_entity, end_entity):
            try:
                # Find highest confidence path (shortest weighted path)
                path = nx.shortest_path(G, start_entity, end_entity, weight='weight')
                
                if len(path) > 1:
                    # Calculate path confidence (minimum edge confidence)
                    path_confidence = 1.0
                    for i in range(len(path) - 1):
                        edge_data = G.get_edge_data(path[i], path[i + 1])
                        edge_confidence = edge_data.get('confidence', 0.0)
                        path_confidence = min(path_confidence, edge_confidence)
                    
                    return {
                        "start": start_entity,
                        "end": end_entity,
                        "path": path,
                        "length": len(path) - 1,
                        "confidence": round(path_confidence, 3)
                    }
                else:
                    # Single entity path
                    return {
                        "start": start_entity,
                        "end": end_entity,
                        "path": path,
                        "length": 0,
                        "confidence": 1.0
                    }
            except nx.NetworkXNoPath:
                return {
                    "start": start_entity,
                    "end": end_entity,
                    "path": [start_entity, end_entity],
                    "length": 0,
                    "confidence": 0.0
                }
        else:
            # No path exists
            return {
                "start": start_entity,
                "end": end_entity,
                "path": [start_entity, end_entity],
                "length": 0,
                "confidence": 0.0
            }
    
    def _find_shortest_path_length(self, kg: KnowledgeGraph, start_entity_id: str, end_entity_id: str) -> Optional[int]:
        """Find the actual shortest path length between two entities in the knowledge graph."""
        import networkx as nx
        
        # Build undirected graph from all relationships (ignoring direction for shortest path)
        G = nx.Graph()
        for rel in kg.relationships:
            G.add_edge(rel.source, rel.target)
        
        # Find shortest path
        if nx.has_path(G, start_entity_id, end_entity_id):
            try:
                shortest_path = nx.shortest_path(G, start_entity_id, end_entity_id)
                return len(shortest_path) - 1  # Number of edges (steps)
            except nx.NetworkXNoPath:
                return None
        else:
            return None
    
    def _generate_weighted_question(
        self,
        kg: KnowledgeGraph,
        template: QuestionTemplate,
        complexity: int
    ) -> Optional[Question]:
        """Generate a weighted reasoning question."""
        
        weighted_paths = kg.metadata.get("weighted_paths", [])
        threshold_queries = kg.metadata.get("threshold_queries", [])
        high_confidence_links = kg.metadata.get("high_confidence_links", [])
        
        if template.variables.get("query_type") == "threshold" and threshold_queries:
            chosen_threshold = self.rng.choice(threshold_queries)
            
            variables = {
                "threshold": str(chosen_threshold["threshold"]),
                "relationship_count": str(chosen_threshold["relationships_above"]),
                "entity_count": str(chosen_threshold["entities_involved"])
            }
            
            answer_value = chosen_threshold["relationships_above"]
            explanation = f"{answer_value} relationships have confidence above {chosen_threshold['threshold']}"
            
        elif template.variables.get("query_type") == "path" and weighted_paths:
            chosen_path = self.rng.choice(weighted_paths)
            
            # CRITICAL FIX: Verify the weighted path actually exists in the knowledge graph relationships
            actual_path_info = self._verify_weighted_path(kg, chosen_path)
            if actual_path_info is None:
                # If path doesn't exist in actual relationships, return None to skip this question
                return None
            
            # Convert entity IDs to names consistently
            path_names = []
            for eid in actual_path_info["path"]:
                entity = kg.get_entity(eid)
                if entity:
                    path_names.append(entity.name)
            
            if not path_names:
                return None
            
            variables = {
                "start_entity": path_names[0],
                "end_entity": path_names[-1],
                "path_confidence": str(actual_path_info["confidence"]),
                "path_length": str(actual_path_info["length"])
            }
            
            if template.answer_type == AnswerType.PATH:
                answer_value = " → ".join(path_names)
                explanation = f"Highest confidence path: {answer_value} (confidence: {actual_path_info['confidence']})"
            else:
                answer_value = actual_path_info["confidence"]
                if actual_path_info["confidence"] > 0:
                    explanation = f"Path confidence: {answer_value}"
                else:
                    explanation = f"No path exists between {path_names[0]} and {path_names[-1]}"
                
        elif high_confidence_links:
            chosen_link = self.rng.choice(high_confidence_links)
            
            variables = {
                "source_entity": chosen_link["source_name"],
                "target_entity": chosen_link["target_name"],
                "confidence": str(chosen_link["confidence"]),
                "relation_type": chosen_link["relation_type"]
            }
            
            answer_value = chosen_link["confidence"]
            explanation = f"Confidence of {chosen_link['relation_type']} relationship: {answer_value}"
            
        else:
            return None
        
        try:
            question_text = template.format_question(variables)
        except (ValueError, KeyError):
            return None
        
        ground_truth = Answer(
            value=answer_value,
            answer_type=template.answer_type,
            explanation=explanation
        )
        
        # Create proper context entities using entity IDs, not names
        context_entity_ids = []
        if template.variables.get("query_type") == "path" and weighted_paths and 'chosen_path' in locals():
            # For path questions, use the actual entity IDs from the chosen path
            context_entity_ids = [chosen_path["path"][0], chosen_path["path"][-1]]  # Start and end entity IDs
        elif high_confidence_links and 'chosen_link' in locals():
            # For relationship questions, use the actual entity IDs
            context_entity_ids = [chosen_link["source"], chosen_link["target"]]
        # For threshold questions, context_entities can remain empty (uses all entities)
        
        return Question(
            id=f"q_{self.rng.randint(100000, 999999)}",
            question_text=question_text,
            question_type=template.question_type,
            complexity_level=complexity,
            ground_truth=ground_truth,
            context_entities=context_entity_ids,
            metadata={"template_id": f"{template.question_type}_{template.complexity_level}"}
        )
    
    def _generate_conflicting_question(
        self,
        kg: KnowledgeGraph,
        template: QuestionTemplate,
        complexity: int
    ) -> Optional[Question]:
        """Generate a conflicting information question."""
        
        detected_conflicts = kg.metadata.get("detected_conflicts", [])
        consistent_subgraphs = kg.metadata.get("consistent_subgraphs", [])
        
        if template.variables.get("query_type") == "detection" and detected_conflicts:
            chosen_conflict = self.rng.choice(detected_conflicts)
            
            # Always use entity names consistently
            source_entity = kg.get_entity(chosen_conflict["source"])
            target_entity = kg.get_entity(chosen_conflict["target"])
            
            if not source_entity or not target_entity:
                return None
            
            source_name = source_entity.name
            target_name = target_entity.name
            
            variables = {
                "source_entity": source_name,
                "target_entity": target_name,
                "conflict_type": chosen_conflict["type"],
                "relation1": chosen_conflict.get("conflicting_relations", ["relation1", "relation2"])[0],
                "relation2": chosen_conflict.get("conflicting_relations", ["relation1", "relation2"])[1] if len(chosen_conflict.get("conflicting_relations", [])) > 1 else "unknown"
            }
            
            if template.answer_type == AnswerType.BOOLEAN:
                answer_value = True  # Conflict exists
                explanation = f"Yes, there is a {chosen_conflict['type']} between {source_name} and {target_name}"
            elif template.answer_type == AnswerType.TEXT:
                answer_value = chosen_conflict["type"]  # Return conflict type as text
                explanation = f"The conflict type is: {answer_value}"
            else:
                answer_value = True
                explanation = f"Yes, there is a {chosen_conflict['type']} between {source_name} and {target_name}"
            
        elif template.variables.get("query_type") == "consistency" and consistent_subgraphs:
            chosen_subgraph = self.rng.choice(consistent_subgraphs)
            
            # For numeric questions about subgraph entities, we need to consider all entities 
            # that have consistent relationships with the named entities in the question
            if template.answer_type == AnswerType.NUMERIC:
                # Use a subset of entities from the subgraph for the question (3 entities)
                question_entities = list(chosen_subgraph["entities"])[:3]
                involved_entities = set(question_entities)  # Start with the named entities
                
                # Find all entities that have consistent relationships with the named entities
                # Only consider "real" entities (not artificial states/categories)
                for rel in kg.relationships:
                    if rel.properties.get("consistent", True):  # Only consider consistent relationships
                        # If relationship connects to one of our named entities, include the other end
                        if rel.source in question_entities:
                            target_entity = kg.get_entity(rel.target)
                            # Only include entities that aren't artificial constructs
                            if target_entity and not target_entity.properties.get("artificial", False):
                                involved_entities.add(rel.target)
                        elif rel.target in question_entities:
                            source_entity = kg.get_entity(rel.source)
                            # Only include entities that aren't artificial constructs  
                            if source_entity and not source_entity.properties.get("artificial", False):
                                involved_entities.add(rel.source)
                
                # Convert to names for display
                display_entity_names = []
                for eid in question_entities:
                    entity = kg.get_entity(eid)
                    if entity:
                        display_entity_names.append(entity.name)
                
                if not display_entity_names:
                    return None
                
                variables = {
                    "entity_set": ", ".join(display_entity_names),
                    "subgraph_size": str(chosen_subgraph["size"]),
                    "relationship_count": str(chosen_subgraph["relationships"])
                }
                
                # Count entities involved in consistent relationships with the named entities
                answer_value = len(involved_entities)
                explanation = f"{answer_value} entities are involved in consistent relationships within the subgraph"
                
            else:
                # For boolean questions, check if the selected entities are actually connected
                # within the consistent subgraph
                question_entities = list(chosen_subgraph["entities"])[:3]
                entity_names = []
                for eid in question_entities:
                    entity = kg.get_entity(eid)
                    if entity:
                        entity_names.append(entity.name)
                
                if not entity_names:
                    return None
                
                variables = {
                    "entity_set": ", ".join(entity_names),
                    "subgraph_size": str(chosen_subgraph["size"]),
                    "relationship_count": str(chosen_subgraph["relationships"])
                }
                
                if template.answer_type == AnswerType.BOOLEAN:
                    # Check if these specific entities have direct or indirect consistent connections
                    # Build a subgraph from only these entities and their consistent relationships
                    entity_subgraph_connected = self._check_entities_connected(kg, question_entities)
                    
                    answer_value = entity_subgraph_connected
                    if entity_subgraph_connected:
                        explanation = f"Yes, the entities {', '.join(entity_names)} are connected through consistent relationships"
                    else:
                        explanation = f"No, the entities {', '.join(entity_names)} are not directly connected through consistent relationships"
                else:
                    # For other answer types, assume consistent subgraph exists
                    answer_value = True
                    explanation = f"Yes, the entities {', '.join(entity_names)} form a consistent subgraph"
            
        else:
            return None
        
        try:
            question_text = template.format_question(variables)
        except (ValueError, KeyError):
            return None
        
        ground_truth = Answer(
            value=answer_value,
            answer_type=template.answer_type,
            explanation=explanation
        )
        
        return Question(
            id=f"q_{self.rng.randint(100000, 999999)}",
            question_text=question_text,
            question_type=template.question_type,
            complexity_level=complexity,
            ground_truth=ground_truth,
            context_entities=question_entities if 'question_entities' in locals() else [v for v in [variables.get("source_entity", ""), variables.get("target_entity", "")] if v],
            metadata={"template_id": f"{template.question_type}_{template.complexity_level}"}
        )


class MultiHopTemplates:
    """Templates for multi-hop reasoning questions."""
    
    def get_templates(self) -> List[QuestionTemplate]:
        """Get all multi-hop question templates."""
        return [
            # Basic path existence
            QuestionTemplate(
                template="Is there a connection between {start_entity} and {end_entity}?",
                question_type=QuestionType.MULTIHOP,
                complexity_level=1,
                required_graph_features=["structured_paths"],
                variables={"start_entity": "Source entity", "end_entity": "Target entity"},
                answer_type=AnswerType.BOOLEAN
            ),
            
            # Path finding
            QuestionTemplate(
                template="What is a shortest path from {start_entity} to {end_entity}?",
                question_type=QuestionType.MULTIHOP,
                complexity_level=2,
                required_graph_features=["structured_paths"],
                variables={"start_entity": "Source entity", "end_entity": "Target entity"},
                answer_type=AnswerType.PATH
            ),
            
            # Intermediate entities
            QuestionTemplate(
                template="What entities lie on the path from {start_entity} to {end_entity}?",
                question_type=QuestionType.MULTIHOP,
                complexity_level=2,
                required_graph_features=["structured_paths"],
                variables={"start_entity": "Source entity", "end_entity": "Target entity"},
                answer_type=AnswerType.ENTITY_LIST
            ),
            
            # Hop counting
            QuestionTemplate(
                template="How many steps does it take to get from {start_entity} to {end_entity}?",
                question_type=QuestionType.MULTIHOP,
                complexity_level=2,
                required_graph_features=["structured_paths"],
                variables={"start_entity": "Source entity", "end_entity": "Target entity"},
                answer_type=AnswerType.NUMERIC
            ),
            
            # Relationship-specific paths
            QuestionTemplate(
                template="Following {relation_types} relationships, how do you get from {start_entity} to {end_entity}?",
                question_type=QuestionType.MULTIHOP,
                complexity_level=3,
                required_graph_features=["structured_paths"],
                variables={"start_entity": "Source entity", "end_entity": "Target entity", "relation_types": "Relationship types"},
                answer_type=AnswerType.PATH
            ),
            
            # Complex reasoning
            QuestionTemplate(
                template="If you start at {start_entity} and can only move through {relation_types} relationships, can you reach {end_entity} in exactly {num_hops} steps?",
                question_type=QuestionType.MULTIHOP,
                complexity_level=4,
                required_graph_features=["structured_paths"],
                variables={"start_entity": "Source entity", "end_entity": "Target entity", "relation_types": "Relationship types", "num_hops": "Number of hops"},
                answer_type=AnswerType.BOOLEAN
            ),
        ]


class HierarchicalTemplates:
    """Templates for hierarchical reasoning questions."""
    
    def get_templates(self) -> List[QuestionTemplate]:
        """Get all hierarchical question templates."""
        return [
            QuestionTemplate(
                template="What is the root category that {leaf_entity} belongs to in the {hierarchy_type} hierarchy?",
                question_type=QuestionType.HIERARCHICAL,
                complexity_level=1,
                required_graph_features=["inheritance_paths"],
                variables={"leaf_entity": "Leaf entity", "hierarchy_type": "Hierarchy type"},
                answer_type=AnswerType.SINGLE_ENTITY
            ),
            
            QuestionTemplate(
                template="Does {leaf_entity} inherit from {root_entity}?",
                question_type=QuestionType.HIERARCHICAL,
                complexity_level=1,
                required_graph_features=["inheritance_paths"],
                variables={"leaf_entity": "Child entity", "root_entity": "Parent entity"},
                answer_type=AnswerType.BOOLEAN
            ),
            
            QuestionTemplate(
                template="Show the complete inheritance path from {leaf_entity} to {root_entity}.",
                question_type=QuestionType.HIERARCHICAL,
                complexity_level=2,
                required_graph_features=["inheritance_paths"],
                variables={"leaf_entity": "Child entity", "root_entity": "Parent entity"},
                answer_type=AnswerType.PATH
            ),
            
            QuestionTemplate(
                template="How many levels of {relation_type} relationships are there between {leaf_entity} and {root_entity}?",
                question_type=QuestionType.HIERARCHICAL,
                complexity_level=2,
                required_graph_features=["inheritance_paths"],
                variables={"leaf_entity": "Child entity", "root_entity": "Parent entity", "relation_type": "Relationship type"},
                answer_type=AnswerType.NUMERIC
            )
        ]


class TemporalTemplates:
    """Templates for temporal reasoning questions."""
    
    def get_templates(self) -> List[QuestionTemplate]:
        """Get all temporal question templates."""
        return [
            QuestionTemplate(
                template="What events occur between {first_event} and {last_event}?",
                question_type=QuestionType.TEMPORAL,
                complexity_level=2,
                required_graph_features=["temporal_sequences"],
                variables={"first_event": "First event", "last_event": "Last event"},
                answer_type=AnswerType.ENTITY_LIST
            ),
            
            QuestionTemplate(
                template="What is the final outcome when {cause_event} triggers a causal chain?",
                question_type=QuestionType.TEMPORAL,
                complexity_level=2,
                required_graph_features=["causal_chains"],
                variables={"cause_event": "Cause event", "sequence_type": "causal"},
                answer_type=AnswerType.SINGLE_ENTITY
            ),
            
            QuestionTemplate(
                template="How many steps are there in the causal chain from {cause_event} to {effect_event}?",
                question_type=QuestionType.TEMPORAL,
                complexity_level=3,
                required_graph_features=["causal_chains"],
                variables={"cause_event": "Cause event", "effect_event": "Effect event", "sequence_type": "causal"},
                answer_type=AnswerType.NUMERIC
            ),
            
            QuestionTemplate(
                template="What is the last event in the temporal sequence starting with {first_event}?",
                question_type=QuestionType.TEMPORAL,
                complexity_level=1,
                required_graph_features=["temporal_sequences"],
                variables={"first_event": "First event"},
                answer_type=AnswerType.SINGLE_ENTITY
            )
        ]


class WeightedTemplates:
    """Templates for weighted reasoning questions."""
    
    def get_templates(self) -> List[QuestionTemplate]:
        """Get all weighted question templates."""
        return [
            QuestionTemplate(
                template="How many relationships have confidence above {threshold}?",
                question_type=QuestionType.WEIGHTED,
                complexity_level=1,
                required_graph_features=["threshold_queries"],
                variables={"threshold": "Confidence threshold", "query_type": "threshold"},
                answer_type=AnswerType.NUMERIC
            ),
            
            QuestionTemplate(
                template="What is the highest confidence path from {start_entity} to {end_entity}?",
                question_type=QuestionType.WEIGHTED,
                complexity_level=2,
                required_graph_features=["weighted_paths"],
                variables={"start_entity": "Start entity", "end_entity": "End entity", "query_type": "path"},
                answer_type=AnswerType.PATH
            ),
            
            QuestionTemplate(
                template="What is the confidence score for the {relation_type} relationship between {source_entity} and {target_entity}?",
                question_type=QuestionType.WEIGHTED,
                complexity_level=1,
                required_graph_features=["high_confidence_links"],
                variables={"source_entity": "Source", "target_entity": "Target", "relation_type": "Relation"},
                answer_type=AnswerType.NUMERIC
            ),
            
            QuestionTemplate(
                template="What is the confidence of the most reliable path from {start_entity} to {end_entity}?",
                question_type=QuestionType.WEIGHTED,
                complexity_level=2,
                required_graph_features=["weighted_paths"],
                variables={"start_entity": "Start entity", "end_entity": "End entity", "query_type": "path"},
                answer_type=AnswerType.NUMERIC
            )
        ]


class ConflictingTemplates:
    """Templates for conflicting information questions."""
    
    def get_templates(self) -> List[QuestionTemplate]:
        """Get all conflicting information question templates."""
        return [
            QuestionTemplate(
                template="Is there a contradiction in the relationship between {source_entity} and {target_entity}?",
                question_type=QuestionType.CONFLICTING,
                complexity_level=2,
                required_graph_features=["detected_conflicts"],
                variables={"source_entity": "Source entity", "target_entity": "Target entity", "query_type": "detection"},
                answer_type=AnswerType.BOOLEAN
            ),
            
            QuestionTemplate(
                template="Are the entities {entity_set} part of a consistent subgraph?",
                question_type=QuestionType.CONFLICTING,
                complexity_level=2,
                required_graph_features=["consistent_subgraphs"],
                variables={"entity_set": "Entity set", "query_type": "consistency"},
                answer_type=AnswerType.BOOLEAN
            ),
            
            QuestionTemplate(
                template="What type of conflict exists between the {relation1} and {relation2} relationships for {source_entity} and {target_entity}?",
                question_type=QuestionType.CONFLICTING,
                complexity_level=3,
                required_graph_features=["detected_conflicts"],
                variables={"source_entity": "Source", "target_entity": "Target", "relation1": "Relation 1", "relation2": "Relation 2", "query_type": "detection"},
                answer_type=AnswerType.TEXT
            ),
            
            QuestionTemplate(
                template="How many entities are involved in consistent relationships within the {entity_set} subgraph?",
                question_type=QuestionType.CONFLICTING,
                complexity_level=2,
                required_graph_features=["consistent_subgraphs"],
                variables={"entity_set": "Entity set", "query_type": "consistency"},
                answer_type=AnswerType.NUMERIC
            )
        ]
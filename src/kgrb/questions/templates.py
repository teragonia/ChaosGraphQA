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
        
        # Create ground truth answer
        if template.answer_type == AnswerType.PATH:
            path_names = [kg.get_entity(eid).name for eid in chosen_path["path"]]
            answer_value = " → ".join(path_names)
            explanation = f"The path from {start_entity.name} to {end_entity.name} is: {answer_value}"
        elif template.answer_type == AnswerType.BOOLEAN:
            answer_value = True  # Path exists
            explanation = f"Yes, there is a path from {start_entity.name} to {end_entity.name} in {chosen_path['length']} hops"
        elif template.answer_type == AnswerType.ENTITY_LIST:
            # Return intermediate entities
            intermediate_entities = [kg.get_entity(eid).name for eid in chosen_path["path"][1:-1]]
            answer_value = intermediate_entities
            explanation = f"Intermediate entities: {', '.join(intermediate_entities)}"
        else:
            answer_value = end_entity.name
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
        
        # Fill template variables
        variables = {
            "leaf_entity": leaf_entity.name,
            "root_entity": root_entity.name,
            "hierarchy_type": chosen_path["hierarchy_type"],
            "relation_type": chosen_path["relation_type"],
            "path_length": str(chosen_path["length"])
        }
        
        # Generate question text
        try:
            question_text = template.format_question(variables)
        except ValueError:
            return None
        
        # Create ground truth answer
        if template.answer_type == AnswerType.BOOLEAN:
            answer_value = True  # Inheritance path exists
            explanation = f"Yes, {leaf_entity.name} inherits from {root_entity.name} through the {chosen_path['hierarchy_type']} hierarchy"
        elif template.answer_type == AnswerType.PATH:
            path_names = [kg.get_entity(eid).name for eid in chosen_path["path"]]
            answer_value = " → ".join(path_names)
            explanation = f"The inheritance path is: {answer_value}"
        else:
            answer_value = root_entity.name
            explanation = f"The root category is {root_entity.name}"
        
        ground_truth = Answer(
            value=answer_value,
            answer_type=template.answer_type,
            explanation=explanation,
            metadata={"inheritance_path": chosen_path}
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
        
        if template.variables.get("sequence_type") == "causal" and causal_chains:
            chosen_data = self.rng.choice(causal_chains)
            variables = {
                "cause_event": kg.get_entity(chosen_data["cause"]).name,
                "effect_event": kg.get_entity(chosen_data["effect"]).name,
                "chain_length": str(chosen_data["length"])
            }
            answer_value = chosen_data["effect"] if template.answer_type == AnswerType.SINGLE_ENTITY else chosen_data["chain"]
            explanation = f"Causal chain: {' → '.join([kg.get_entity(eid).name for eid in chosen_data['chain']])}"
            context_entity_ids = chosen_data["chain"][:2]  # Use first two entity IDs from causal chain
            
        elif temporal_sequences:
            chosen_sequence = self.rng.choice(temporal_sequences)
            event_names = chosen_sequence["event_names"]
            event_ids = chosen_sequence["events"]  # Get entity IDs, not names
            
            variables = {
                "first_event": event_names[0],
                "last_event": event_names[-1],
                "sequence_length": str(chosen_sequence["length"]),
                "start_time": chosen_sequence["start_time"],
                "end_time": chosen_sequence["end_time"]
            }
            
            if template.answer_type == AnswerType.ENTITY_LIST:
                answer_value = event_names[1:-1]  # Intermediate events
                explanation = f"Events between {event_names[0]} and {event_names[-1]}: {', '.join(answer_value)}"
            else:
                answer_value = event_names[-1]
                explanation = f"The last event in the sequence is {event_names[-1]}"
            
            context_entity_ids = event_ids  # Use all relevant entity IDs from sequence
            
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
            path_names = [kg.get_entity(eid).name for eid in chosen_path["path"]]
            
            variables = {
                "start_entity": path_names[0],
                "end_entity": path_names[-1],
                "path_confidence": str(chosen_path["confidence"]),
                "path_length": str(chosen_path["length"])
            }
            
            if template.answer_type == AnswerType.PATH:
                answer_value = " → ".join(path_names)
                explanation = f"Highest confidence path: {answer_value} (confidence: {chosen_path['confidence']})"
            else:
                answer_value = chosen_path["confidence"]
                explanation = f"Path confidence: {answer_value}"
                
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
        
        return Question(
            id=f"q_{self.rng.randint(100000, 999999)}",
            question_text=question_text,
            question_type=template.question_type,
            complexity_level=complexity,
            ground_truth=ground_truth,
            context_entities=[v for v in [variables.get("start_entity", ""), variables.get("end_entity", "")] if v],
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
            
            source_name = kg.get_entity(chosen_conflict["source"]).name if kg.get_entity(chosen_conflict["source"]) else "Unknown"
            target_name = kg.get_entity(chosen_conflict["target"]).name if kg.get_entity(chosen_conflict["target"]) else "Unknown"
            
            variables = {
                "source_entity": source_name,
                "target_entity": target_name,
                "conflict_type": chosen_conflict["type"],
                "relation1": chosen_conflict.get("conflicting_relations", ["relation1", "relation2"])[0],
                "relation2": chosen_conflict.get("conflicting_relations", ["relation1", "relation2"])[1] if len(chosen_conflict.get("conflicting_relations", [])) > 1 else "unknown"
            }
            
            answer_value = True  # Conflict exists
            explanation = f"Yes, there is a {chosen_conflict['type']} between {source_name} and {target_name}"
            
        elif template.variables.get("query_type") == "consistency" and consistent_subgraphs:
            chosen_subgraph = self.rng.choice(consistent_subgraphs)
            entity_names = [kg.get_entity(eid).name for eid in chosen_subgraph["entities"][:3]]  # First 3 entities
            
            variables = {
                "entity_set": ", ".join(entity_names),
                "subgraph_size": str(chosen_subgraph["size"]),
                "relationship_count": str(chosen_subgraph["relationships"])
            }
            
            answer_value = True  # Subgraph is consistent
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
            context_entities=[v for v in [variables.get("source_entity", ""), variables.get("target_entity", "")] if v],
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
                template="What is the shortest path from {start_entity} to {end_entity}?",
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
"""Answer validation utilities for ChaosGraphQA."""

import re
from typing import Any, Dict, List, Optional, Union
from ..models.question import Answer, AnswerType, Question
from ..models.graph import KnowledgeGraph
from ..evaluators.graph_algorithms import GraphAlgorithms


class AnswerValidator:
    """Validates and scores LLM responses against ground truth answers."""
    
    def __init__(self):
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._graph_algorithms: Optional[GraphAlgorithms] = None
    
    def set_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """Set the knowledge graph for advanced validation."""
        self._knowledge_graph = kg
        self._graph_algorithms = GraphAlgorithms(kg)
    
    def validate_response(
        self,
        question: Question,
        llm_response: str,
        strict: bool = False
    ) -> Dict[str, Any]:
        """Validate an LLM response against ground truth.
        
        Args:
            question: The question with ground truth answer
            llm_response: The LLM's response text
            strict: Whether to use strict matching
            
        Returns:
            Dictionary with validation results
        """
        ground_truth = question.ground_truth
        
        # Parse LLM response based on expected answer type
        parsed_response = self._parse_response(llm_response, ground_truth.answer_type)
        
        # Compare with ground truth
        is_correct, score, explanation = self._compare_answers(
            parsed_response,
            ground_truth.value,
            ground_truth.answer_type,
            strict,
            question
        )
        
        return {
            "is_correct": is_correct,
            "score": score,
            "explanation": explanation,
            "parsed_response": parsed_response,
            "ground_truth": ground_truth.value,
            "answer_type": ground_truth.answer_type
        }
    
    def _parse_response(self, response: str, answer_type: AnswerType) -> Any:
        """Parse LLM response based on structured format."""
        response = response.strip()
        
        # Extract the answer from "ANSWER: [content]" format
        answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        if not answer_match:
            # Fallback: try to parse the entire response
            answer_content = response
        else:
            answer_content = answer_match.group(1).strip()
        
        if answer_type == AnswerType.BOOLEAN:
            return self._parse_structured_boolean(answer_content)
        elif answer_type == AnswerType.NUMERIC:
            return self._parse_structured_numeric(answer_content)
        elif answer_type == AnswerType.SINGLE_ENTITY:
            return self._parse_structured_entity(answer_content)
        elif answer_type == AnswerType.ENTITY_LIST:
            return self._parse_structured_entity_list(answer_content)
        elif answer_type == AnswerType.PATH:
            return self._parse_structured_path(answer_content)
        elif answer_type == AnswerType.TEXT:
            return self._parse_structured_text(answer_content)
        else:
            return answer_content
    
    def _parse_structured_boolean(self, answer_content: str) -> Optional[bool]:
        """Parse boolean from structured format."""
        answer_content = answer_content.strip().lower()
        
        if answer_content == "yes":
            return True
        elif answer_content == "no":
            return False
        
        return None
    
    def _parse_structured_numeric(self, answer_content: str) -> Optional[float]:
        """Parse numeric from structured format."""
        try:
            return float(answer_content.strip())
        except ValueError:
            return None
    
    def _parse_structured_entity(self, answer_content: str) -> str:
        """Parse single entity from structured format."""
        return answer_content.strip()
    
    def _parse_structured_entity_list(self, answer_content: str) -> List[str]:
        """Parse entity list from structured format: [Entity1, Entity2, Entity3]"""
        answer_content = answer_content.strip()
        
        # Handle empty list
        if answer_content == "[]":
            return []
        
        # Remove brackets if present
        if answer_content.startswith('[') and answer_content.endswith(']'):
            answer_content = answer_content[1:-1].strip()
        
        # Handle empty content
        if not answer_content:
            return []
        
        # Split by commas and clean each entity
        entities = []
        for entity in answer_content.split(','):
            entity = entity.strip()
            if entity:
                entities.append(entity)
        
        return entities
    
    def _parse_structured_path(self, answer_content: str) -> str:
        """Parse path from structured format: [Entity1 → Entity2 → Entity3]"""
        answer_content = answer_content.strip()
        
        # Remove brackets if present
        if answer_content.startswith('[') and answer_content.endswith(']'):
            answer_content = answer_content[1:-1].strip()
        
        return answer_content
    
    def _parse_structured_text(self, answer_content: str) -> str:
        """Parse text from structured format."""
        return answer_content.strip()
    
    def _compare_answers(
        self,
        parsed_response: Any,
        ground_truth: Any,
        answer_type: AnswerType,
        strict: bool,
        question: Optional[Question] = None
    ) -> tuple[bool, float, str]:
        """Compare parsed response with ground truth."""
        
        if answer_type == AnswerType.BOOLEAN:
            return self._compare_boolean(parsed_response, ground_truth, strict)
        elif answer_type == AnswerType.NUMERIC:
            return self._compare_numeric(parsed_response, ground_truth, strict)
        elif answer_type == AnswerType.SINGLE_ENTITY:
            return self._compare_entity(parsed_response, ground_truth, strict)
        elif answer_type == AnswerType.ENTITY_LIST:
            return self._compare_entity_list(parsed_response, ground_truth, strict)
        elif answer_type == AnswerType.PATH:
            return self._compare_path(parsed_response, ground_truth, strict, question)
        else:
            return self._compare_text(parsed_response, ground_truth, strict)
    
    def _compare_boolean(self, response: Optional[bool], truth: bool, strict: bool) -> tuple[bool, float, str]:
        """Compare boolean values."""
        if response is None:
            return False, 0.0, "Could not parse boolean response"
        
        if response == truth:
            return True, 1.0, "Correct boolean answer"
        else:
            return False, 0.0, f"Incorrect: expected {truth}, got {response}"
    
    def _compare_numeric(self, response: Optional[float], truth: Any, strict: bool) -> tuple[bool, float, str]:
        """Compare numeric values."""
        if response is None:
            return False, 0.0, "Could not parse numeric response"
        
        # Convert truth to float if it's a string
        try:
            if isinstance(truth, str):
                truth_val = float(truth)
            else:
                truth_val = float(truth)
        except (ValueError, TypeError):
            return False, 0.0, f"Invalid ground truth value: {truth}"
        
        tolerance = 0.001 if strict else 0.1
        if abs(response - truth_val) <= tolerance:
            return True, 1.0, "Correct numeric answer"
        else:
            return False, 0.0, f"Incorrect: expected {truth_val}, got {response}"
    
    def _compare_entity(self, response: str, truth: str, strict: bool) -> tuple[bool, float, str]:
        """Compare entity names."""
        if strict:
            is_correct = response.lower() == truth.lower()
        else:
            # Fuzzy matching - check if main parts match
            response_clean = re.sub(r'[^\w\s]', '', response.lower())
            truth_clean = re.sub(r'[^\w\s]', '', truth.lower())
            is_correct = response_clean == truth_clean or response_clean in truth_clean or truth_clean in response_clean
        
        score = 1.0 if is_correct else 0.0
        explanation = "Correct entity" if is_correct else f"Incorrect: expected '{truth}', got '{response}'"
        
        return is_correct, score, explanation
    
    def _compare_entity_list(self, response: List[str], truth: List[str], strict: bool) -> tuple[bool, float, str]:
        """Compare entity lists."""
        if not response:
            return False, 0.0, "Empty response"
        
        # Calculate partial credit
        correct_entities = 0
        total_entities = len(truth)
        
        for truth_entity in truth:
            for response_entity in response:
                if self._compare_entity(response_entity, truth_entity, strict)[0]:
                    correct_entities += 1
                    break
        
        score = correct_entities / total_entities if total_entities > 0 else 0.0
        is_correct = score >= 0.8  # Allow some tolerance
        
        explanation = f"Found {correct_entities}/{total_entities} correct entities"
        
        return is_correct, score, explanation
    
    def _compare_path(self, response: str, truth: str, strict: bool, question: Optional[Question] = None) -> tuple[bool, float, str]:
        """Compare path responses with multi-hop awareness."""
        # Extract entity names from both paths
        response_entities = re.findall(r'[A-Z][a-zA-Z]+', response)
        truth_entities = re.findall(r'[A-Z][a-zA-Z]+', truth)
        
        if not response_entities:
            return False, 0.0, "No entities found in path response"
        
        # For multi-hop questions with graph algorithms available, validate if response is a valid shortest path
        if (question and question.question_type == "multihop" and 
            "shortest path" in question.question_text.lower() and
            self._knowledge_graph and self._graph_algorithms and len(response_entities) >= 2):
            
            # Map entity names to IDs
            start_entity_id = self._find_entity_id_by_name(response_entities[0])
            end_entity_id = self._find_entity_id_by_name(response_entities[-1])
            
            if start_entity_id and end_entity_id:
                # Check if the LLM's response is a valid shortest path
                all_shortest_paths = self._graph_algorithms.find_all_shortest_paths(start_entity_id, end_entity_id)
                
                if all_shortest_paths:
                    # Convert response entity names to IDs for comparison
                    response_path_ids = []
                    for entity_name in response_entities:
                        entity_id = self._find_entity_id_by_name(entity_name)
                        if entity_id:
                            response_path_ids.append(entity_id)
                    
                    # Check if the response path matches any valid shortest path
                    if response_path_ids in all_shortest_paths:
                        return True, 1.0, f"Valid shortest path found (one of {len(all_shortest_paths)} valid paths)"
                    
                    # Check if it's a valid path but not shortest
                    if self._graph_algorithms.verify_path_validity(response_path_ids):
                        actual_shortest_length = len(all_shortest_paths[0]) - 1 if all_shortest_paths else 0
                        response_length = len(response_path_ids) - 1
                        if response_length > actual_shortest_length:
                            return False, 0.5, f"Valid path but not shortest: {response_length} hops vs {actual_shortest_length} hops"
                    
                    return False, 0.0, f"Invalid path: does not match any of the {len(all_shortest_paths)} valid shortest paths"
        
        # Fallback to original comparison logic for non-multi-hop questions
        # Check if paths match (allowing for minor variations)
        if len(response_entities) != len(truth_entities):
            # Partial credit based on overlapping entities
            overlap = len(set(response_entities) & set(truth_entities))
            score = overlap / len(truth_entities) if truth_entities else 0.0
            return False, score, f"Path length mismatch: expected {len(truth_entities)} entities, got {len(response_entities)}"
        
        # Check entity-by-entity match
        correct_positions = 0
        for i, (resp_entity, truth_entity) in enumerate(zip(response_entities, truth_entities)):
            if self._compare_entity(resp_entity, truth_entity, strict)[0]:
                correct_positions += 1
        
        score = correct_positions / len(truth_entities)
        is_correct = score >= 0.8
        
        explanation = f"Path entities match: {correct_positions}/{len(truth_entities)}"
        
        return is_correct, score, explanation
    
    def _find_entity_id_by_name(self, entity_name: str) -> Optional[str]:
        """Find entity ID by name in the knowledge graph."""
        if not self._knowledge_graph:
            return None
        
        # Direct name match
        for entity_id, entity in self._knowledge_graph.entities.items():
            if entity.name.lower() == entity_name.lower():
                return entity_id
        
        # Partial name match as fallback
        for entity_id, entity in self._knowledge_graph.entities.items():
            if entity_name.lower() in entity.name.lower() or entity.name.lower() in entity_name.lower():
                return entity_id
        
        return None
    
    def _compare_text(self, response: str, truth: str, strict: bool) -> tuple[bool, float, str]:
        """Compare text responses."""
        if strict:
            is_correct = response.lower().strip() == truth.lower().strip()
        else:
            # Check for key word overlap
            response_words = set(response.lower().split())
            truth_words = set(truth.lower().split())
            
            if len(truth_words) == 0:
                is_correct = len(response_words) == 0
            else:
                overlap = len(response_words & truth_words) / len(truth_words)
                is_correct = overlap >= 0.6
        
        score = 1.0 if is_correct else 0.0
        explanation = "Text match" if is_correct else "Text mismatch"
        
        return is_correct, score, explanation
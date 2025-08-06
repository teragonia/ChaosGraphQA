"""Ground truth verification system for KGRB."""

from typing import Dict, List, Optional, Any, Set, Tuple, Union
from ..models.graph import KnowledgeGraph
from ..models.question import Question, QuestionType, Answer, AnswerType
from .graph_algorithms import GraphAlgorithms


class GroundTruthVerifier:
    """Main ground truth verification system."""
    
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.algorithms = GraphAlgorithms(kg)
    
    def verify_question_answer(self, question: Question) -> Dict[str, Any]:
        """Verify that a question's ground truth answer is correct."""
        
        verification_result = {
            "question_id": question.id,
            "question_type": question.question_type,
            "is_valid": False,
            "confidence": 0.0,
            "verification_method": "unknown",
            "details": {},
            "alternative_answers": []
        }
        
        try:
            if question.question_type == QuestionType.MULTIHOP:
                result = self._verify_multihop_answer(question)
            elif question.question_type == QuestionType.HIERARCHICAL:
                result = self._verify_hierarchical_answer(question)
            elif question.question_type == QuestionType.TEMPORAL:
                result = self._verify_temporal_answer(question)
            elif question.question_type == QuestionType.WEIGHTED:
                result = self._verify_weighted_answer(question)
            elif question.question_type == QuestionType.CONFLICTING:
                result = self._verify_conflicting_answer(question)
            else:
                result = {"is_valid": False, "confidence": 0.0, "details": {"error": "Unsupported question type"}}
            
            verification_result.update(result)
            
        except Exception as e:
            verification_result.update({
                "is_valid": False,
                "confidence": 0.0,
                "details": {"error": str(e)}
            })
        
        return verification_result
    
    def _verify_multihop_answer(self, question: Question) -> Dict[str, Any]:
        """Verify multi-hop reasoning answers."""
        answer = question.ground_truth
        context_entities = question.context_entities
        
        if len(context_entities) < 2:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "multihop_path_verification",
                "details": {"error": "Insufficient context entities for multi-hop question"}
            }
        
        start_entity = context_entities[0]
        end_entity = context_entities[1]
        
        # Verify entities exist
        if start_entity not in self.kg.entities or end_entity not in self.kg.entities:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "multihop_entity_check",
                "details": {"error": "Context entities not found in graph"}
            }
        
        if answer.answer_type == AnswerType.BOOLEAN:
            return self._verify_path_existence(start_entity, end_entity, answer.value)
        
        elif answer.answer_type == AnswerType.PATH:
            return self._verify_path_answer(start_entity, end_entity, answer.value)
        
        elif answer.answer_type == AnswerType.NUMERIC:
            return self._verify_path_length(start_entity, end_entity, answer.value)
        
        elif answer.answer_type == AnswerType.ENTITY_LIST:
            return self._verify_intermediate_entities(start_entity, end_entity, answer.value)
        
        else:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "multihop_answer_type_check",
                "details": {"error": f"Unsupported answer type for multihop: {answer.answer_type}"}
            }
    
    def _verify_path_existence(self, start: str, end: str, expected: bool) -> Dict[str, Any]:
        """Verify boolean path existence answer."""
        actual_exists = self.algorithms.path_exists(start, end)
        
        is_valid = actual_exists == expected
        confidence = 1.0 if is_valid else 0.0
        
        details = {
            "expected_exists": expected,
            "actual_exists": actual_exists,
            "verification_method": "networkx_has_path"
        }
        
        if actual_exists:
            shortest_path = self.algorithms.find_shortest_path(start, end)
            details["shortest_path_length"] = len(shortest_path) - 1 if shortest_path else None
            details["shortest_path"] = shortest_path
        
        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "verification_method": "path_existence",
            "details": details
        }
    
    def _verify_path_answer(self, start: str, end: str, expected_path: str) -> Dict[str, Any]:
        """Verify path string answer."""
        # Parse expected path
        expected_entities = self._parse_path_string(expected_path)
        
        if not expected_entities or len(expected_entities) < 2:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "path_parsing",
                "details": {"error": "Could not parse expected path"}
            }
        
        # Verify path is valid
        path_valid = self.algorithms.verify_path_validity(expected_entities)
        
        if not path_valid:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "path_validity_check",
                "details": {
                    "expected_path": expected_entities,
                    "error": "Expected path is not valid in graph"
                }
            }
        
        # Find actual shortest path for comparison
        actual_shortest = self.algorithms.find_shortest_path(start, end)
        
        details = {
            "expected_path": expected_entities,
            "actual_shortest_path": actual_shortest,
            "path_valid": path_valid
        }
        
        # Check if expected path is optimal (for shortest path questions)
        is_optimal = actual_shortest and len(expected_entities) == len(actual_shortest)
        
        # Find alternative paths for context
        all_paths = self.algorithms.find_all_simple_paths(start, end, max_length=len(expected_entities) + 2)
        details["alternative_paths_count"] = len(all_paths)
        details["is_optimal_length"] = is_optimal
        
        confidence = 1.0 if path_valid else 0.0
        
        return {
            "is_valid": path_valid,
            "confidence": confidence,
            "verification_method": "path_answer_verification",
            "details": details
        }
    
    def _verify_path_length(self, start: str, end: str, expected_length: float) -> Dict[str, Any]:
        """Verify numeric path length answer."""
        actual_length = self.algorithms.get_path_length(start, end)
        
        if actual_length is None:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "path_length_check",
                "details": {"error": "No path exists between entities"}
            }
        
        is_valid = abs(actual_length - expected_length) < 0.001
        confidence = 1.0 if is_valid else 0.0
        
        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "verification_method": "path_length_verification",
            "details": {
                "expected_length": expected_length,
                "actual_length": actual_length
            }
        }
    
    def _verify_intermediate_entities(self, start: str, end: str, expected_entities: List[str]) -> Dict[str, Any]:
        """Verify intermediate entities answer."""
        shortest_path = self.algorithms.find_shortest_path(start, end)
        
        if not shortest_path:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "intermediate_entities_path_check",
                "details": {"error": "No path exists between entities"}
            }
        
        # Get actual intermediate entities (exclude start and end)
        actual_intermediate = shortest_path[1:-1] if len(shortest_path) > 2 else []
        
        # Convert entity IDs to names for comparison
        actual_names = [self.kg.get_entity(eid).name for eid in actual_intermediate if self.kg.get_entity(eid)]
        
        # Check if expected entities match actual intermediate entities
        expected_set = set(expected_entities)
        actual_set = set(actual_names)
        
        intersection = expected_set & actual_set
        precision = len(intersection) / len(expected_set) if expected_set else 1.0
        recall = len(intersection) / len(actual_set) if actual_set else 1.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        is_valid = f1_score >= 0.8  # Allow some tolerance
        
        return {
            "is_valid": is_valid,
            "confidence": f1_score,
            "verification_method": "intermediate_entities_verification",
            "details": {
                "expected_entities": expected_entities,
                "actual_entities": actual_names,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "shortest_path": shortest_path
            }
        }
    
    def _parse_path_string(self, path_string: str) -> List[str]:
        """Parse path string into list of entity IDs."""
        # Handle arrow notation
        if '→' in path_string:
            entity_names = [name.strip() for name in path_string.split('→')]
        elif '->' in path_string:
            entity_names = [name.strip() for name in path_string.split('->')]
        else:
            # Try to extract entity names
            import re
            entity_names = re.findall(r'[A-Z][a-zA-Z]+', path_string)
        
        # Convert names to entity IDs
        entity_ids = []
        for name in entity_names:
            # Find entity with matching name
            for entity_id, entity in self.kg.entities.items():
                if entity.name.lower() == name.lower():
                    entity_ids.append(entity_id)
                    break
        
        return entity_ids
    
    def _verify_hierarchical_answer(self, question: Question) -> Dict[str, Any]:
        """Verify hierarchical reasoning answers."""
        # Placeholder for hierarchical verification
        return {
            "is_valid": False,
            "confidence": 0.0,
            "verification_method": "hierarchical_verification",
            "details": {"error": "Hierarchical verification not yet implemented"}
        }
    
    def _verify_temporal_answer(self, question: Question) -> Dict[str, Any]:
        """Verify temporal reasoning answers."""
        # Placeholder for temporal verification
        return {
            "is_valid": False,
            "confidence": 0.0,
            "verification_method": "temporal_verification",
            "details": {"error": "Temporal verification not yet implemented"}
        }
    
    def _verify_weighted_answer(self, question: Question) -> Dict[str, Any]:
        """Verify weighted reasoning answers."""
        # Placeholder for weighted verification
        return {
            "is_valid": False,
            "confidence": 0.0,
            "verification_method": "weighted_verification",
            "details": {"error": "Weighted verification not yet implemented"}
        }
    
    def _verify_conflicting_answer(self, question: Question) -> Dict[str, Any]:
        """Verify conflicting information answers."""
        # Placeholder for conflict verification
        return {
            "is_valid": False,
            "confidence": 0.0,
            "verification_method": "conflict_verification",
            "details": {"error": "Conflict verification not yet implemented"}
        }
    
    def verify_question_set(self, questions: List[Question]) -> Dict[str, Any]:
        """Verify all questions in a set."""
        results = []
        total_valid = 0
        total_confidence = 0.0
        
        for question in questions:
            result = self.verify_question_answer(question)
            results.append(result)
            
            if result["is_valid"]:
                total_valid += 1
            total_confidence += result["confidence"]
        
        return {
            "total_questions": len(questions),
            "valid_questions": total_valid,
            "validity_rate": total_valid / len(questions) if questions else 0.0,
            "average_confidence": total_confidence / len(questions) if questions else 0.0,
            "individual_results": results
        }
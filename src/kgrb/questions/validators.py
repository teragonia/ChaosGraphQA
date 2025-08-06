"""Answer validation utilities for KGRB."""

import re
from typing import Any, Dict, List, Optional, Union
from ..models.question import Answer, AnswerType, Question


class AnswerValidator:
    """Validates and scores LLM responses against ground truth answers."""
    
    def __init__(self):
        pass
    
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
            strict
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
        """Parse LLM response based on expected answer type."""
        response = response.strip()
        
        if answer_type == AnswerType.BOOLEAN:
            return self._parse_boolean(response)
        elif answer_type == AnswerType.NUMERIC:
            return self._parse_numeric(response)
        elif answer_type == AnswerType.SINGLE_ENTITY:
            return self._parse_entity(response)
        elif answer_type == AnswerType.ENTITY_LIST:
            return self._parse_entity_list(response)
        elif answer_type == AnswerType.PATH:
            return self._parse_path(response)
        else:
            return response
    
    def _parse_boolean(self, response: str) -> Optional[bool]:
        """Parse boolean response."""
        response_lower = response.lower()
        
        # Look for explicit yes/no
        if any(word in response_lower for word in ["yes", "true", "correct", "exists"]):
            return True
        elif any(word in response_lower for word in ["no", "false", "incorrect", "doesn't exist", "does not exist"]):
            return False
        
        return None
    
    def _parse_numeric(self, response: str) -> Optional[float]:
        """Parse numeric response."""
        # Extract first number from response
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        return None
    
    def _parse_entity(self, response: str) -> str:
        """Parse single entity response."""
        # Take first meaningful word/phrase
        # Remove common prefixes
        response = re.sub(r'^(the answer is|it is|that would be|the entity is)[\s:]*', '', response, flags=re.IGNORECASE)
        
        # Extract first capitalized word or phrase in quotes
        quoted = re.search(r'["\']([^"\']+)["\']', response)
        if quoted:
            return quoted.group(1)
        
        # Extract first word that looks like an entity name
        words = response.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                return word
        
        return response.split('.')[0].strip()
    
    def _parse_entity_list(self, response: str) -> List[str]:
        """Parse list of entities."""
        entities = []
        
        # Look for comma-separated or bullet-pointed lists
        if ',' in response:
            parts = response.split(',')
        elif '•' in response or '*' in response:
            parts = re.split(r'[•*]\s*', response)
        elif '\n' in response:
            parts = response.split('\n')
        else:
            # Try to extract multiple capitalized words
            parts = re.findall(r'\b[A-Z][a-zA-Z]+\b', response)
        
        for part in parts:
            entity = self._parse_entity(part.strip())
            if entity and len(entity) > 1:
                entities.append(entity)
        
        return entities
    
    def _parse_path(self, response: str) -> str:
        """Parse path response."""
        # Look for arrow notation or "to" connections
        if '→' in response or '->' in response:
            # Already in arrow format
            return response.strip()
        
        # Look for "to" or "→" patterns
        path_patterns = [
            r'([A-Z][a-zA-Z]+)(?:\s+(?:to|→|->) \s*([A-Z][a-zA-Z]+))+',
            r'([A-Z][a-zA-Z]+)(?:\s*→\s*([A-Z][a-zA-Z]+))+'
        ]
        
        for pattern in path_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(0)
        
        return response.strip()
    
    def _compare_answers(
        self,
        parsed_response: Any,
        ground_truth: Any,
        answer_type: AnswerType,
        strict: bool
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
            return self._compare_path(parsed_response, ground_truth, strict)
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
    
    def _compare_path(self, response: str, truth: str, strict: bool) -> tuple[bool, float, str]:
        """Compare path responses."""
        # Extract entity names from both paths
        response_entities = re.findall(r'[A-Z][a-zA-Z]+', response)
        truth_entities = re.findall(r'[A-Z][a-zA-Z]+', truth)
        
        if not response_entities:
            return False, 0.0, "No entities found in path response"
        
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
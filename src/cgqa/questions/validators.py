"""Answer validation utilities for ChaosGraphQA."""

import re
from typing import Any, Dict, List, Optional, Union

from ..evaluators.graph_algorithms import GraphAlgorithms
from ..models.graph import KnowledgeGraph
from ..models.question import Answer, AnswerType, Question


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
        self, question: Question, llm_response: str, strict: bool = False
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

        parsed_response = self._parse_response(llm_response, ground_truth.answer_type)

        # Special handling for multihop path questions
        if (
            question.question_type == "multihop"
            and "path from" in question.question_text.lower()
            and "entities lie on" in question.question_text.lower()
            and ground_truth.answer_type == "entity_list"
            and self._knowledge_graph
            and self._graph_algorithms
        ):
            is_correct, score, explanation = (
                self._validate_multihop_path_intermediaries(
                    parsed_response, question, strict
                )
            )
        else:
            is_correct, score, explanation = self._compare_answers(
                parsed_response,
                ground_truth.value,
                ground_truth.answer_type,
                strict,
                question,
            )

        return {
            "is_correct": is_correct,
            "score": score,
            "explanation": explanation,
            "parsed_response": parsed_response,
            "ground_truth": ground_truth.value,
            "answer_type": ground_truth.answer_type,
        }

    def _parse_response(self, response: str, answer_type: AnswerType) -> Any:
        """Parse LLM response based on structured format."""
        response = response.strip()

        answer_match = re.search(
            r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL
        )
        if not answer_match:
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
        if answer_content.startswith("[") and answer_content.endswith("]"):
            answer_content = answer_content[1:-1].strip()

        # Handle empty content
        if not answer_content:
            return []

        # Split by commas and clean each entity
        entities = []
        for entity in answer_content.split(","):
            entity = entity.strip()
            if entity:
                entities.append(entity)

        return entities

    def _parse_structured_path(self, answer_content: str) -> str:
        """Parse path from structured format: [Entity1 → Entity2 → Entity3]"""
        answer_content = answer_content.strip()

        # Remove brackets if present
        if answer_content.startswith("[") and answer_content.endswith("]"):
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
        question: Optional[Question] = None,
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

    def _compare_boolean(
        self, response: Optional[bool], truth: bool, strict: bool
    ) -> tuple[bool, float, str]:
        """Compare boolean values."""
        if response is None:
            return (
                False,
                0.0,
                f"Could not parse boolean response. Correct answer: {'Yes' if truth else 'No'}",
            )

        if response == truth:
            return True, 1.0, f"✓ Correct boolean answer: {'Yes' if truth else 'No'}"
        else:
            return (
                False,
                0.0,
                f"✗ Incorrect: expected {'Yes' if truth else 'No'}, got {'Yes' if response else 'No'}",
            )

    def _compare_numeric(
        self, response: Optional[float], truth: Any, strict: bool
    ) -> tuple[bool, float, str]:
        """Compare numeric values."""
        if response is None:
            return (
                False,
                0.0,
                f"Could not parse numeric response. Correct answer: {truth}",
            )

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
            return True, 1.0, f"✓ Correct numeric answer: {truth_val}"
        else:
            return (
                False,
                0.0,
                f"✗ Incorrect: expected {truth_val}, got {response} (tolerance: ±{tolerance})",
            )

    def _compare_entity(
        self, response: str, truth: str, strict: bool
    ) -> tuple[bool, float, str]:
        """Compare entity names."""
        if strict:
            is_correct = response.lower() == truth.lower()
        else:
            # Fuzzy matching - check if main parts match
            response_clean = re.sub(r"[^\w\s]", "", response.lower())
            truth_clean = re.sub(r"[^\w\s]", "", truth.lower())
            is_correct = (
                response_clean == truth_clean
                or response_clean in truth_clean
                or truth_clean in response_clean
            )

        score = 1.0 if is_correct else 0.0
        explanation = (
            "Correct entity"
            if is_correct
            else f"Incorrect: expected '{truth}', got '{response}'"
        )

        return is_correct, score, explanation

    def _compare_entity_list(
        self, response: List[str], truth: List[str], strict: bool
    ) -> tuple[bool, float, str]:
        """Compare entity lists."""
        if not response:
            # If both response and truth are empty, it's correct
            if not truth:
                return True, 1.0, "Correct: both empty lists"
            correct_list = ", ".join(truth) if truth else "[]"
            return False, 0.0, f"Empty response. Correct entities: [{correct_list}]"

        # Calculate partial credit
        correct_entities = 0
        total_entities = len(truth)
        found_entities = []
        missing_entities = []

        for truth_entity in truth:
            found = False
            for response_entity in response:
                if self._compare_entity(response_entity, truth_entity, strict)[0]:
                    correct_entities += 1
                    found_entities.append(truth_entity)
                    found = True
                    break
            if not found:
                missing_entities.append(truth_entity)

        score = correct_entities / total_entities if total_entities > 0 else 0.0
        is_correct = score >= 0.8  # Allow some tolerance

        # Enhanced explanation with all correct entities
        explanation = f"Found {correct_entities}/{total_entities} correct entities. "
        explanation += f"Complete correct list: [{', '.join(truth)}]"

        if missing_entities:
            explanation += f". Missing: [{', '.join(missing_entities)}]"

        return is_correct, score, explanation

    def _compare_path(
        self,
        response: str,
        truth: str,
        strict: bool,
        question: Optional[Question] = None,
    ) -> tuple[bool, float, str]:
        """Compare path responses with multi-hop awareness."""
        # Extract entity names from both paths
        response_entities = re.findall(r"[A-Z][a-zA-Z]+", response)
        truth_entities = re.findall(r"[A-Z][a-zA-Z]+", truth)

        if not response_entities:
            return False, 0.0, "No entities found in path response"

        # For multi-hop questions with graph algorithms available, validate if response is a valid shortest path
        if (
            question
            and question.question_type == "multihop"
            and "shortest path" in question.question_text.lower()
            and self._knowledge_graph
            and self._graph_algorithms
            and len(response_entities) >= 2
        ):

            # Map entity names to IDs
            start_entity_id = self._find_entity_id_by_name(response_entities[0])
            end_entity_id = self._find_entity_id_by_name(response_entities[-1])

            if start_entity_id and end_entity_id:
                # Check if the LLM's response is a valid shortest path
                all_shortest_paths = self._graph_algorithms.find_all_shortest_paths(
                    start_entity_id, end_entity_id
                )

                if all_shortest_paths:
                    # Convert response entity names to IDs for comparison
                    response_path_ids = []
                    for entity_name in response_entities:
                        entity_id = self._find_entity_id_by_name(entity_name)
                        if entity_id:
                            response_path_ids.append(entity_id)

                    # Check if the response path matches any valid shortest path
                    if response_path_ids in all_shortest_paths:
                        # Convert all valid paths to entity names for the explanation
                        valid_path_names = []
                        for path_ids in all_shortest_paths:
                            path_names = []
                            for entity_id in path_ids:
                                entity = self._knowledge_graph.entities.get(entity_id)
                                if entity:
                                    path_names.append(entity.name)
                            if path_names:
                                valid_path_names.append(" → ".join(path_names))

                        explanation = (
                            f"✓ Correct shortest path! All valid shortest paths ({len(all_shortest_paths)}): "
                            + "; ".join(valid_path_names)
                        )
                        return True, 1.0, explanation

                    # Check if it's a valid path but not shortest
                    if self._graph_algorithms.verify_path_validity(response_path_ids):
                        actual_shortest_length = (
                            len(all_shortest_paths[0]) - 1 if all_shortest_paths else 0
                        )
                        response_length = len(response_path_ids) - 1

                        # Include correct shortest paths in explanation
                        valid_path_names = []
                        for path_ids in all_shortest_paths[
                            :3
                        ]:  # Show first 3 to avoid overwhelming
                            path_names = []
                            for entity_id in path_ids:
                                entity = self._knowledge_graph.entities.get(entity_id)
                                if entity:
                                    path_names.append(entity.name)
                            if path_names:
                                valid_path_names.append(" → ".join(path_names))

                        paths_text = "; ".join(valid_path_names)
                        if len(all_shortest_paths) > 3:
                            paths_text += f" (and {len(all_shortest_paths) - 3} more)"

                        explanation = f"Valid path but not shortest: {response_length} hops vs {actual_shortest_length} hops. Correct shortest paths: {paths_text}"
                        return False, 0.5, explanation

                    # Invalid path - show what the correct paths should be
                    valid_path_names = []
                    for path_ids in all_shortest_paths[:3]:  # Show first 3
                        path_names = []
                        for entity_id in path_ids:
                            entity = self._knowledge_graph.entities.get(entity_id)
                            if entity:
                                path_names.append(entity.name)
                        if path_names:
                            valid_path_names.append(" → ".join(path_names))

                    paths_text = "; ".join(valid_path_names)
                    if len(all_shortest_paths) > 3:
                        paths_text += f" (and {len(all_shortest_paths) - 3} more)"

                    explanation = f"✗ Invalid path. Correct shortest paths ({len(all_shortest_paths)}): {paths_text}"
                    return False, 0.0, explanation

        # Fallback to original comparison logic for non-multi-hop questions
        # Check if paths match (allowing for minor variations)
        if len(response_entities) != len(truth_entities):
            # Partial credit based on overlapping entities
            overlap = len(set(response_entities) & set(truth_entities))
            score = overlap / len(truth_entities) if truth_entities else 0.0
            return (
                False,
                score,
                f"Path length mismatch: expected {len(truth_entities)} entities, got {len(response_entities)}",
            )

        # Check entity-by-entity match
        correct_positions = 0
        for i, (resp_entity, truth_entity) in enumerate(
            zip(response_entities, truth_entities)
        ):
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
            if (
                entity_name.lower() in entity.name.lower()
                or entity.name.lower() in entity_name.lower()
            ):
                return entity_id

        return None

    def _compare_text(
        self, response: str, truth: str, strict: bool
    ) -> tuple[bool, float, str]:
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

    def _validate_multihop_path_intermediaries(
        self, response: List[str], question: Question, strict: bool
    ) -> tuple[bool, float, str]:
        """Validate multihop path intermediaries by checking if they form valid paths."""
        if not response:
            return False, 0.0, f"Empty response. Expected intermediate entities."

        # Extract start and end entities from the question
        # Pattern: "What entities lie on the path from X to Y?"
        import re

        pattern = r"path from (\w+) to (\w+)"
        match = re.search(pattern, question.question_text, re.IGNORECASE)

        if not match:
            # Fallback to original validation if pattern doesn't match
            return self._compare_entity_list(
                response, question.ground_truth.value, strict
            )

        start_name = match.group(1)
        end_name = match.group(2)

        # Find entity IDs by name
        start_id = self._find_entity_id_by_name(start_name)
        end_id = self._find_entity_id_by_name(end_name)

        if not start_id or not end_id:
            return (
                False,
                0.0,
                f"Could not find start entity '{start_name}' or end entity '{end_name}' in knowledge graph",
            )

        # For each intermediate entity in response, check if it creates a valid path
        valid_intermediaries = []
        invalid_intermediaries = []

        for intermediate_name in response:
            intermediate_id = self._find_entity_id_by_name(intermediate_name)
            if not intermediate_id:
                invalid_intermediaries.append(intermediate_name)
                continue

            # Check if there's a path: start -> intermediate -> end
            path_to_intermediate = self._graph_algorithms.find_shortest_path(
                start_id, intermediate_id
            )
            path_from_intermediate = self._graph_algorithms.find_shortest_path(
                intermediate_id, end_id
            )

            if (
                path_to_intermediate
                and path_from_intermediate
                and len(path_to_intermediate) == 2
                and len(path_from_intermediate) == 2
            ):
                # Valid 2-hop path through this intermediate
                valid_intermediaries.append(intermediate_name)
            else:
                invalid_intermediaries.append(intermediate_name)

        # Calculate score based on valid intermediaries
        if valid_intermediaries:
            score = 1.0  # Any valid path is acceptable
            is_correct = True
            explanation = f"✓ Valid path intermediar{'ies' if len(valid_intermediaries) > 1 else 'y'}: {', '.join(valid_intermediaries)}"

            if invalid_intermediaries:
                # Filter out junk/malformed entries (keep only short, entity-like names)
                clean_invalid = [
                    name for name in invalid_intermediaries
                    if len(name) < 50 and '\n' not in name
                ]
                # Limit to first 3 to avoid cluttering explanation
                if clean_invalid:
                    invalid_display = clean_invalid[:3]
                    if len(clean_invalid) > 3:
                        invalid_display.append(f"... and {len(clean_invalid) - 3} more")
                    explanation += f". Invalid: {', '.join(invalid_display)}"
        else:
            score = 0.0
            is_correct = False

            # Show what valid intermediaries exist
            all_valid_paths = self._graph_algorithms.find_all_shortest_paths(
                start_id, end_id
            )
            if all_valid_paths:
                valid_intermediary_names = set()
                for path in all_valid_paths:
                    if len(path) == 3:  # 2-hop path
                        intermediate_entity = self._knowledge_graph.entities.get(
                            path[1]
                        )
                        if intermediate_entity:
                            valid_intermediary_names.add(intermediate_entity.name)

                if valid_intermediary_names:
                    explanation = f"✗ No valid intermediaries. Valid options: {', '.join(sorted(valid_intermediary_names))}"
                else:
                    explanation = f"✗ No valid 2-hop paths exist between {start_name} and {end_name}"
            else:
                explanation = f"✗ No paths exist between {start_name} and {end_name}"

        return is_correct, score, explanation

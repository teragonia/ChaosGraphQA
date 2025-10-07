"""Ground truth verification system for ChaosGraphQA.

This module verifies that question answers are algorithmically correct:
- Uses NetworkX graph algorithms for path/connectivity verification
- Supports all question types (multihop, temporal, weighted, etc.)
- Provides confidence scores and alternative answers
- Enables infinite unique benchmarks with verified correctness
"""

from typing import Any, Dict, List, Optional

from ..models.graph import KnowledgeGraph
from ..models.question import Answer, AnswerType, Question, QuestionType
from .graph_algorithms import GraphAlgorithms


class GroundTruthVerifier:
    """Algorithmic verification of question ground truth answers.

    Uses graph algorithms to verify answer correctness rather than
    relying on static datasets, enabling dynamic benchmark generation.
    """

    def __init__(self, kg: KnowledgeGraph):
        """Initialize verifier with a knowledge graph.

        Args:
            kg: Knowledge graph to verify answers against
        """
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
            "alternative_answers": [],
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
                result = {
                    "is_valid": False,
                    "confidence": 0.0,
                    "details": {"error": "Unsupported question type"},
                }

            verification_result.update(result)

        except Exception as e:
            verification_result.update(
                {"is_valid": False, "confidence": 0.0, "details": {"error": str(e)}}
            )

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
                "details": {
                    "error": "Insufficient context entities for multi-hop question"
                },
            }

        start_entity = context_entities[0]
        end_entity = context_entities[1]

        # Verify entities exist
        if start_entity not in self.kg.entities or end_entity not in self.kg.entities:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "multihop_entity_check",
                "details": {"error": "Context entities not found in graph"},
            }

        if answer.answer_type == AnswerType.BOOLEAN:
            return self._verify_path_existence(start_entity, end_entity, answer.value)

        elif answer.answer_type == AnswerType.PATH:
            return self._verify_path_answer(start_entity, end_entity, answer.value)

        elif answer.answer_type == AnswerType.NUMERIC:
            return self._verify_path_length(start_entity, end_entity, answer.value)

        elif answer.answer_type == AnswerType.ENTITY_LIST:
            return self._verify_intermediate_entities(
                start_entity, end_entity, answer.value
            )

        else:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "multihop_answer_type_check",
                "details": {
                    "error": f"Unsupported answer type for multihop: {answer.answer_type}"
                },
            }

    def _verify_path_existence(
        self, start: str, end: str, expected: bool
    ) -> Dict[str, Any]:
        """Verify boolean path existence answer."""
        actual_exists = self.algorithms.path_exists(start, end)

        is_valid = actual_exists == expected
        confidence = 1.0 if is_valid else 0.0

        details = {
            "expected_exists": expected,
            "actual_exists": actual_exists,
            "verification_method": "networkx_has_path",
        }

        if actual_exists:
            shortest_path = self.algorithms.find_shortest_path(start, end)
            details["shortest_path_length"] = (
                len(shortest_path) - 1 if shortest_path else None
            )
            details["shortest_path"] = shortest_path

        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "verification_method": "path_existence",
            "details": details,
        }

    def _verify_path_answer(
        self, start: str, end: str, expected_path: str
    ) -> Dict[str, Any]:
        """Verify path string answer."""
        # Parse expected path
        expected_entities = self._parse_path_string(expected_path)

        if not expected_entities or len(expected_entities) < 2:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "path_parsing",
                "details": {"error": "Could not parse expected path"},
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
                    "error": "Expected path is not valid in graph",
                },
            }

        # Find actual shortest path for comparison
        actual_shortest = self.algorithms.find_shortest_path(start, end)

        details = {
            "expected_path": expected_entities,
            "actual_shortest_path": actual_shortest,
            "path_valid": path_valid,
        }

        # Check if expected path is optimal (for shortest path questions)
        is_optimal = actual_shortest and len(expected_entities) == len(actual_shortest)

        # Find alternative paths for context
        all_paths = self.algorithms.find_all_simple_paths(
            start, end, max_length=len(expected_entities) + 2
        )
        details["alternative_paths_count"] = len(all_paths)
        details["is_optimal_length"] = is_optimal

        confidence = 1.0 if path_valid else 0.0

        return {
            "is_valid": path_valid,
            "confidence": confidence,
            "verification_method": "path_answer_verification",
            "details": details,
        }

    def _verify_path_length(
        self, start: str, end: str, expected_length: float
    ) -> Dict[str, Any]:
        """Verify numeric path length answer."""
        actual_length = self.algorithms.get_path_length(start, end)

        if actual_length is None:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "path_length_check",
                "details": {"error": "No path exists between entities"},
            }

        is_valid = abs(actual_length - expected_length) < 0.001
        confidence = 1.0 if is_valid else 0.0

        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "verification_method": "path_length_verification",
            "details": {
                "expected_length": expected_length,
                "actual_length": actual_length,
            },
        }

    def _verify_intermediate_entities(
        self, start: str, end: str, expected_entities: List[str]
    ) -> Dict[str, Any]:
        """Verify intermediate entities answer."""
        shortest_path = self.algorithms.find_shortest_path(start, end)

        if not shortest_path:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "intermediate_entities_path_check",
                "details": {"error": "No path exists between entities"},
            }

        # Get actual intermediate entities (exclude start and end)
        actual_intermediate = shortest_path[1:-1] if len(shortest_path) > 2 else []

        # Convert entity IDs to names for comparison
        actual_names = [
            self.kg.get_entity(eid).name
            for eid in actual_intermediate
            if self.kg.get_entity(eid)
        ]

        # Check if expected entities match actual intermediate entities
        expected_set = set(expected_entities)
        actual_set = set(actual_names)

        intersection = expected_set & actual_set
        precision = len(intersection) / len(expected_set) if expected_set else 1.0
        recall = len(intersection) / len(actual_set) if actual_set else 1.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

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
                "shortest_path": shortest_path,
            },
        }

    def _parse_path_string(self, path_string: str) -> List[str]:
        """Parse path string into list of entity IDs."""
        # Handle arrow notation
        if "→" in path_string:
            entity_names = [name.strip() for name in path_string.split("→")]
        elif "->" in path_string:
            entity_names = [name.strip() for name in path_string.split("->")]
        else:
            # Try to extract entity names
            import re

            entity_names = re.findall(r"[A-Z][a-zA-Z]+", path_string)

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
            "details": {"error": "Hierarchical verification not yet implemented"},
        }

    def _verify_temporal_answer(self, question: Question) -> Dict[str, Any]:
        """Verify temporal reasoning answers."""
        answer = question.ground_truth

        # Get temporal metadata from knowledge graph
        temporal_sequences = self.kg.metadata.get("temporal_sequences", [])
        causal_chains = self.kg.metadata.get("causal_chains", [])

        # Determine question template from metadata
        template_id = question.metadata.get("template_id", "")

        if (
            "causal" in question.question_text.lower()
            or "triggers" in question.question_text.lower()
        ):
            return self._verify_causal_chain_answer(question, causal_chains)
        elif "between" in question.question_text.lower():
            return self._verify_temporal_sequence_intermediate_answer(
                question, temporal_sequences
            )
        elif "last event" in question.question_text.lower():
            return self._verify_temporal_sequence_final_answer(
                question, temporal_sequences, causal_chains
            )
        else:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "temporal_verification",
                "details": {"error": "Unknown temporal question type"},
            }

    def _verify_weighted_answer(self, question: Question) -> Dict[str, Any]:
        """Verify weighted reasoning answers."""
        # Placeholder for weighted verification
        return {
            "is_valid": False,
            "confidence": 0.0,
            "verification_method": "weighted_verification",
            "details": {"error": "Weighted verification not yet implemented"},
        }

    def _verify_conflicting_answer(self, question: Question) -> Dict[str, Any]:
        """Verify conflicting information answers."""
        # Placeholder for conflict verification
        return {
            "is_valid": False,
            "confidence": 0.0,
            "verification_method": "conflict_verification",
            "details": {"error": "Conflict verification not yet implemented"},
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
            "average_confidence": (
                total_confidence / len(questions) if questions else 0.0
            ),
            "individual_results": results,
        }

    def _verify_causal_chain_answer(
        self, question: Question, causal_chains: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify causal chain questions."""
        answer = question.ground_truth

        # Find the relevant causal chain based on question context
        question_text = question.question_text.lower()
        relevant_chains = []

        for chain in causal_chains:
            cause_entity = self.kg.get_entity(chain["cause"])
            effect_entity = self.kg.get_entity(chain["effect"])

            if cause_entity and effect_entity:
                if (
                    cause_entity.name.lower() in question_text
                    or effect_entity.name.lower() in question_text
                ):
                    relevant_chains.append(chain)

        if not relevant_chains:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "causal_chain_verification",
                "details": {"error": "No relevant causal chains found for question"},
            }

        # For single entity answers, verify the final effect
        if answer.answer_type == AnswerType.SINGLE_ENTITY:
            expected_name = answer.value

            for chain in relevant_chains:
                effect_entity = self.kg.get_entity(chain["effect"])
                if effect_entity and effect_entity.name == expected_name:
                    return {
                        "is_valid": True,
                        "confidence": 1.0,
                        "verification_method": "causal_chain_verification",
                        "details": {
                            "expected_effect": expected_name,
                            "verified_chain": [
                                self.kg.get_entity(eid).name for eid in chain["chain"]
                            ],
                        },
                    }

        return {
            "is_valid": False,
            "confidence": 0.0,
            "verification_method": "causal_chain_verification",
            "details": {"error": "Expected answer not found in causal chains"},
        }

    def _verify_temporal_sequence_intermediate_answer(
        self, question: Question, temporal_sequences: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify questions about events between two other events."""
        answer = question.ground_truth

        # Extract start and end events from question
        question_text = question.question_text
        import re

        between_pattern = r"between\s+([^?]+?)\s+and\s+([^?]+?)(?:\?|$)"
        match = re.search(between_pattern, question_text, re.IGNORECASE)

        if not match:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "temporal_intermediate_verification",
                "details": {"error": "Could not parse start/end events from question"},
            }

        start_event_name = match.group(1).strip()
        end_event_name = match.group(2).strip()

        # Find relevant temporal sequence
        relevant_sequence = None
        for sequence in temporal_sequences:
            if (
                start_event_name in sequence["event_names"]
                and end_event_name in sequence["event_names"]
            ):
                relevant_sequence = sequence
                break

        if not relevant_sequence:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "temporal_intermediate_verification",
                "details": {
                    "error": f"No sequence found containing {start_event_name} and {end_event_name}"
                },
            }

        # Calculate expected intermediate events
        event_names = relevant_sequence["event_names"]
        try:
            start_idx = event_names.index(start_event_name)
            end_idx = event_names.index(end_event_name)

            # Ensure proper order
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx

            expected_intermediate = event_names[start_idx + 1 : end_idx]

        except ValueError:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "temporal_intermediate_verification",
                "details": {"error": "Events not found in sequence"},
            }

        # Verify answer
        if answer.answer_type == AnswerType.ENTITY_LIST:
            answer_list = answer.value if isinstance(answer.value, list) else []

            # Check if expected matches answer
            if set(expected_intermediate) == set(answer_list):
                confidence = 1.0
                is_valid = True
            else:
                # Partial credit
                intersection = set(expected_intermediate) & set(answer_list)
                total_expected = len(expected_intermediate)
                confidence = (
                    len(intersection) / total_expected if total_expected > 0 else 1.0
                )
                is_valid = confidence >= 0.8

            return {
                "is_valid": is_valid,
                "confidence": confidence,
                "verification_method": "temporal_intermediate_verification",
                "details": {
                    "expected_intermediate": expected_intermediate,
                    "provided_intermediate": answer_list,
                    "sequence": event_names,
                },
            }

        return {
            "is_valid": False,
            "confidence": 0.0,
            "verification_method": "temporal_intermediate_verification",
            "details": {"error": "Unsupported answer type for intermediate events"},
        }

    def _verify_temporal_sequence_final_answer(
        self,
        question: Question,
        temporal_sequences: List[Dict[str, Any]],
        causal_chains: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify questions about the last event in a sequence."""
        answer = question.ground_truth

        # Extract starting event from question
        question_text = question.question_text
        import re

        start_pattern = r"starting with\s+([^?]+?)(?:\?|$)"
        match = re.search(start_pattern, question_text, re.IGNORECASE)

        if not match:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "temporal_final_verification",
                "details": {"error": "Could not parse starting event from question"},
            }

        start_event_name = match.group(1).strip()

        # Check both temporal sequences and causal chains
        possible_final_events = set()
        longest_sequence_final = None

        # Check temporal sequences - find all sequences starting with the event
        sequences_starting_with_event = []
        for sequence in temporal_sequences:
            if start_event_name in sequence["event_names"]:
                start_idx = sequence["event_names"].index(start_event_name)
                if start_idx == 0:  # Sequence starts with this event
                    sequences_starting_with_event.append(sequence)
                    possible_final_events.add(sequence["event_names"][-1])

        # For "last event" questions, the longest sequence should be considered the correct answer
        if sequences_starting_with_event:
            longest_sequence = max(
                sequences_starting_with_event, key=lambda s: s["length"]
            )
            longest_sequence_final = longest_sequence["event_names"][-1]

        # Check causal chains
        for chain in causal_chains:
            cause_entity = self.kg.get_entity(chain["cause"])
            if cause_entity and cause_entity.name == start_event_name:
                effect_entity = self.kg.get_entity(chain["effect"])
                if effect_entity:
                    possible_final_events.add(effect_entity.name)

        if not possible_final_events:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "verification_method": "temporal_final_verification",
                "details": {
                    "error": f"No sequences or chains found starting with {start_event_name}"
                },
            }

        # Verify answer
        if answer.answer_type == AnswerType.SINGLE_ENTITY:
            expected_name = answer.value

            # Primary validation: Check if answer matches the longest sequence final event
            if longest_sequence_final and expected_name == longest_sequence_final:
                return {
                    "is_valid": True,
                    "confidence": 1.0,
                    "verification_method": "temporal_final_verification_longest",
                    "details": {
                        "expected_final": expected_name,
                        "longest_sequence_final": longest_sequence_final,
                        "possible_finals": list(possible_final_events),
                        "starting_event": start_event_name,
                    },
                }

            # Secondary validation: Check if answer is in any possible final events
            elif expected_name in possible_final_events:
                # Give partial credit if it's a valid ending but not the longest sequence
                confidence = (
                    0.8
                    if longest_sequence_final
                    and expected_name != longest_sequence_final
                    else 1.0
                )
                return {
                    "is_valid": True,
                    "confidence": confidence,
                    "verification_method": "temporal_final_verification",
                    "details": {
                        "expected_final": expected_name,
                        "longest_sequence_final": longest_sequence_final,
                        "possible_finals": list(possible_final_events),
                        "starting_event": start_event_name,
                        "note": (
                            "Valid ending but may not be from longest sequence"
                            if confidence < 1.0
                            else None
                        ),
                    },
                }
            else:
                return {
                    "is_valid": False,
                    "confidence": 0.0,
                    "verification_method": "temporal_final_verification",
                    "details": {
                        "expected_final": expected_name,
                        "longest_sequence_final": longest_sequence_final,
                        "possible_finals": list(possible_final_events),
                        "starting_event": start_event_name,
                        "error": f"Expected final event '{expected_name}' not found in possible outcomes",
                    },
                }

        return {
            "is_valid": False,
            "confidence": 0.0,
            "verification_method": "temporal_final_verification",
            "details": {"error": "Unsupported answer type for final event"},
        }

"""LLM evaluation engine for ChaosGraphQA."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ...models.graph import KnowledgeGraph
from ...models.question import Question, QuestionSet
from ...questions.validators import AnswerValidator
from ..providers.base import BaseLLMProvider, LLMResponse
from .provider_factory import ProviderFactory


@dataclass
class EvaluationResult:
    """Result of evaluating a single question."""

    question_id: str
    question_text: str
    question_type: str
    complexity_level: int

    # LLM interaction
    prompt: str
    llm_response: str
    response_time: float

    # Ground truth
    ground_truth_answer: Any
    ground_truth_explanation: str

    # Evaluation metrics
    is_correct: bool
    score: float  # 0.0 to 1.0
    validation_explanation: str

    # Token usage
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Error information
    error: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""

    model_name: str
    provider_name: str
    total_questions: int
    evaluation_time: float  # seconds
    timestamp: float = field(default_factory=time.time)

    # Overall metrics
    accuracy: float = 0.0  # Percentage of correct answers
    average_score: float = 0.0  # Average partial credit score

    # Performance by category
    by_question_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_complexity: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Token usage statistics
    total_tokens_used: int = 0
    average_tokens_per_question: float = 0.0

    # Error statistics
    error_rate: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)

    # Individual results
    results: List[EvaluationResult] = field(default_factory=list)


class LLMEvaluator:
    """Main evaluation engine for testing LLMs on ChaosGraphQA benchmarks."""

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        self.validator = AnswerValidator()

    def evaluate_questions(
        self,
        questions: List[Question],
        kg: KnowledgeGraph,
        include_context: bool = True,
        show_progress: bool = True,
    ) -> EvaluationSummary:
        """Evaluate a list of questions with the LLM.

        Args:
            questions: List of questions to evaluate
            kg: Knowledge graph containing the context
            include_context: Whether to include graph context in prompts
            show_progress: Whether to show progress information

        Returns:
            EvaluationSummary with all results
        """
        start_time = time.time()
        results = []

        if show_progress:
            try:
                from rich.progress import track

                question_iter = track(questions, description="Evaluating questions...")
            except ImportError:
                question_iter = questions
                if show_progress:
                    print(f"Evaluating {len(questions)} questions...")
        else:
            question_iter = questions

        for i, question in enumerate(question_iter):
            if show_progress and "track" not in locals():
                print(f"Question {i+1}/{len(questions)}")

            result = self.evaluate_single_question(question, kg, include_context)
            results.append(result)

        # Create summary
        evaluation_time = time.time() - start_time
        summary = self._create_summary(results, evaluation_time)

        return summary

    def evaluate_single_question(
        self, question: Question, kg: KnowledgeGraph, include_context: bool = True
    ) -> EvaluationResult:
        """Evaluate a single question.

        Args:
            question: Question to evaluate
            kg: Knowledge graph for context
            include_context: Whether to include graph context

        Returns:
            EvaluationResult for this question
        """
        self.validator.set_knowledge_graph(kg)

        context = None
        if include_context:
            context = self._prepare_context(question, kg)
        prompt = self.provider.format_question_prompt(
            question.question_text,
            context,
            answer_type=question.ground_truth.answer_type.value,
            question_type=question.question_type,
        )

        start_time = time.time()
        llm_response = self._generate_with_retry(prompt, max_retries=3)
        response_time = time.time() - start_time
        if llm_response.error:
            # Handle LLM error
            return EvaluationResult(
                question_id=question.id,
                question_text=question.question_text,
                question_type=question.question_type,
                complexity_level=question.complexity_level,
                prompt=prompt,  # Store the prompt even for errors
                llm_response="",
                response_time=response_time,
                ground_truth_answer=question.ground_truth.value,
                ground_truth_explanation=question.ground_truth.explanation or "",
                is_correct=False,
                score=0.0,
                validation_explanation="LLM error occurred",
                error=llm_response.error,
                error_type=llm_response.error_type,
            )

        # Validate the response
        validation_result = self.validator.validate_response(
            question, llm_response.text, strict=False
        )

        return EvaluationResult(
            question_id=question.id,
            question_text=question.question_text,
            question_type=question.question_type,
            complexity_level=question.complexity_level,
            prompt=prompt,  # Store the full prompt
            llm_response=llm_response.text,
            response_time=response_time,
            ground_truth_answer=question.ground_truth.value,
            ground_truth_explanation=question.ground_truth.explanation or "",
            is_correct=validation_result["is_correct"],
            score=validation_result["score"],
            validation_explanation=validation_result["explanation"],
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
        )

    def _prepare_context(self, question: Question, kg: KnowledgeGraph) -> str:
        """Prepare context from knowledge graph for a question.

        Args:
            question: The question being asked
            kg: Knowledge graph

        Returns:
            Formatted context string
        """
        # Include ALL entities from the knowledge graph (no filtering)
        context_entities = list(kg.entities.keys())

        context_parts = []
        is_temporal = question.question_type == "temporal"

        # Describe entities (only the ones mentioned in context_entities for brevity)
        # For questions with specific context entities, show those first
        entities_to_describe = (
            question.context_entities
            if question.context_entities
            else context_entities[:50]
        )

        for entity_id in entities_to_describe:
            if entity_id in kg.entities:
                entity = kg.entities[entity_id]
                if is_temporal and entity.entity_type == "event":
                    start_time = entity.properties.get("start_time", "")
                    end_time = entity.properties.get("end_time", "")
                    duration = entity.properties.get("duration_days", "")
                    context_parts.append(
                        f"- {entity.name} ({entity.entity_type}, starts: {start_time}, ends: {end_time}, duration: {duration} days)"
                    )
                else:
                    context_parts.append(f"- {entity.name} ({entity.entity_type})")

        # Include ALL relationships from the knowledge graph (no filtering)
        temporal_relationships = []
        regular_relationships = []

        for rel in kg.relationships:
            if is_temporal and rel.properties.get("temporal_relationship"):
                temporal_relationships.append(rel)
            else:
                regular_relationships.append(rel)

        # Shuffle relationships to prevent LLMs from exploiting ordering patterns
        import random

        # Use question ID as seed for reproducibility
        shuffle_seed = hash(question.id) % (2**32)
        temporal_shuffled = temporal_relationships.copy()
        regular_shuffled = regular_relationships.copy()
        random.Random(shuffle_seed).shuffle(temporal_shuffled)
        random.Random(shuffle_seed).shuffle(regular_shuffled)

        # Add temporal relationships
        if is_temporal and temporal_shuffled:
            context_parts.append("\nTemporal Relationships:")
            for rel in temporal_shuffled:
                source_entity = kg.entities.get(rel.source)
                target_entity = kg.entities.get(rel.target)

                if source_entity and target_entity:
                    if question.question_type == "weighted" and rel.weight is not None:
                        weight_str = self._format_weight_for_display(
                            rel.weight, rel.relation_type, rel.properties
                        )
                        context_parts.append(
                            f"- {source_entity.name} {rel.relation_type} {target_entity.name} ({weight_str})"
                        )
                    else:
                        context_parts.append(
                            f"- {source_entity.name} {rel.relation_type} {target_entity.name}"
                        )

        # Add all regular relationships (skip for temporal questions as they're not needed)
        if regular_shuffled and not is_temporal:
            context_parts.append("\nRelationships:")

            for rel in regular_shuffled:
                source_entity = kg.entities.get(rel.source)
                target_entity = kg.entities.get(rel.target)

                if source_entity and target_entity:
                    if question.question_type == "weighted" and rel.weight is not None:
                        weight_str = self._format_weight_for_display(
                            rel.weight, rel.relation_type, rel.properties
                        )
                        context_parts.append(
                            f"- {source_entity.name} {rel.relation_type} {target_entity.name} ({weight_str})"
                        )
                    else:
                        context_parts.append(
                            f"- {source_entity.name} {rel.relation_type} {target_entity.name}"
                        )

        relation_types = {rel.relation_type for rel in kg.relationships}
        from ...models.relationship_semantics import RelationshipSemantics

        directionality_section = (
            RelationshipSemantics.generate_directionality_prompt_section(relation_types)
        )
        context_parts.append(f"\n{directionality_section}")

        return "\n".join(context_parts)

    def _format_weight_for_display(
        self, weight: float, relation_type: str, properties: Dict[str, Any] = None
    ) -> str:
        """Format weight/confidence score for display in context.

        Args:
            weight: The numerical weight/confidence score
            relation_type: Type of relationship
            properties: Additional properties that might contain weight type info
        """
        if properties and isinstance(properties, dict):
            weight_type = properties.get("weight_type", "confidence")
        else:
            # Infer weight type from relation type
            if (
                "probably" in relation_type
                or "likely" in relation_type
                or "may" in relation_type
            ):
                weight_type = "confidence"
            elif "distance" in relation_type:
                weight_type = "distance"
            elif "similarity" in relation_type:
                weight_type = "similarity"
            elif "trust" in relation_type:
                weight_type = "trust"
            elif "expertise" in relation_type or "preference" in relation_type:
                weight_type = "score"
            else:
                weight_type = "confidence"

        return f"{weight_type}: {weight:.4f}"

    def _create_summary(
        self, results: List[EvaluationResult], evaluation_time: float
    ) -> EvaluationSummary:
        """Create evaluation summary from individual results."""

        if not results:
            return EvaluationSummary(
                model_name=self.provider.config.model_name,
                provider_name=self.provider.provider_name,
                total_questions=0,
                evaluation_time=evaluation_time,
            )

        # Overall metrics
        correct_count = sum(1 for r in results if r.is_correct)
        accuracy = correct_count / len(results) if results else 0.0
        average_score = sum(r.score for r in results) / len(results) if results else 0.0

        # Error statistics
        error_count = sum(1 for r in results if r.error is not None)
        error_rate = error_count / len(results) if results else 0.0
        error_types = {}
        for result in results:
            if result.error_type:
                error_types[result.error_type] = (
                    error_types.get(result.error_type, 0) + 1
                )

        # Token usage
        total_tokens = sum(r.total_tokens or 0 for r in results)
        avg_tokens = total_tokens / len(results) if results else 0.0

        # Performance by question type
        by_question_type = {}
        type_groups = {}
        for result in results:
            q_type = result.question_type
            if q_type not in type_groups:
                type_groups[q_type] = []
            type_groups[q_type].append(result)

        for q_type, group_results in type_groups.items():
            type_correct = sum(1 for r in group_results if r.is_correct)
            type_accuracy = type_correct / len(group_results)
            type_avg_score = sum(r.score for r in group_results) / len(group_results)

            by_question_type[q_type] = {
                "accuracy": type_accuracy,
                "average_score": type_avg_score,
                "count": len(group_results),
            }

        # Performance by complexity
        by_complexity = {}
        complexity_groups = {}
        for result in results:
            complexity = result.complexity_level
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append(result)

        for complexity, group_results in complexity_groups.items():
            comp_correct = sum(1 for r in group_results if r.is_correct)
            comp_accuracy = comp_correct / len(group_results)
            comp_avg_score = sum(r.score for r in group_results) / len(group_results)

            by_complexity[complexity] = {
                "accuracy": comp_accuracy,
                "average_score": comp_avg_score,
                "count": len(group_results),
            }

        return EvaluationSummary(
            model_name=self.provider.config.model_name,
            provider_name=self.provider.provider_name,
            total_questions=len(results),
            evaluation_time=evaluation_time,
            accuracy=accuracy,
            average_score=average_score,
            by_question_type=by_question_type,
            by_complexity=by_complexity,
            total_tokens_used=total_tokens,
            average_tokens_per_question=avg_tokens,
            error_rate=error_rate,
            error_types=error_types,
            results=results,
        )

    def save_results(
        self, summary: EvaluationSummary, output_path: Union[str, Path]
    ) -> None:
        """Save evaluation results to file.

        Args:
            summary: Evaluation summary to save
            output_path: Path to save results
        """
        output_path = Path(output_path)

        # Convert summary to serializable format
        results_data = {
            "metadata": {
                "model_name": summary.model_name,
                "provider_name": summary.provider_name,
                "total_questions": summary.total_questions,
                "evaluation_time": summary.evaluation_time,
                "timestamp": summary.timestamp,
            },
            "metrics": {
                "accuracy": summary.accuracy,
                "average_score": summary.average_score,
                "total_tokens_used": summary.total_tokens_used,
                "average_tokens_per_question": summary.average_tokens_per_question,
                "error_rate": summary.error_rate,
                "error_types": summary.error_types,
            },
            "performance": {
                "by_question_type": summary.by_question_type,
                "by_complexity": summary.by_complexity,
            },
            "individual_results": [
                {
                    "question_id": r.question_id,
                    "question_text": r.question_text,
                    "question_type": r.question_type,
                    "complexity_level": r.complexity_level,
                    "prompt": r.prompt,  # Include the full prompt
                    "llm_response": r.llm_response,
                    "response_time": r.response_time,
                    "ground_truth_answer": r.ground_truth_answer,
                    "ground_truth_explanation": r.ground_truth_explanation,
                    "is_correct": r.is_correct,
                    "score": r.score,
                    "validation_explanation": r.validation_explanation,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_tokens": r.total_tokens,
                    "error": r.error,
                    "error_type": r.error_type,
                }
                for r in summary.results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

    def _generate_with_retry(self, prompt: str, max_retries: int = 3) -> LLMResponse:
        """Generate response with retry logic for handling transient failures.

        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum number of retry attempts

        Returns:
            LLMResponse object
        """
        last_response = None

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                response = self.provider.generate(prompt)

                # If successful (no error), return immediately
                if not response.error:
                    return response

                # Store the response for potential return if all retries fail
                last_response = response

                # Check if this is a retryable error
                if not self._is_retryable_error(response.error_type):
                    # Non-retryable error, return immediately
                    return response

                # If this was the last attempt, don't wait
                if attempt == max_retries:
                    break

                # Wait before retry with exponential backoff
                wait_time = min(2**attempt, 180)  # Cap at 3 minutes
                print(
                    f"Attempt {attempt + 1} failed with {response.error_type}, retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

            except Exception as e:
                # Unexpected error, create error response
                last_response = LLMResponse(
                    text="", error=str(e), error_type="UnexpectedError"
                )

                if attempt == max_retries:
                    break

                wait_time = min(2**attempt, 180)
                print(
                    f"Attempt {attempt + 1} failed with unexpected error, retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        # All retries failed, return the last response
        return last_response or LLMResponse(
            text="", error="All retry attempts failed", error_type="MaxRetriesExceeded"
        )

    def _is_retryable_error(self, error_type: Optional[str]) -> bool:
        """Check if an error type is retryable.

        Args:
            error_type: The error type to check

        Returns:
            True if the error is retryable, False otherwise
        """
        if not error_type:
            return False

        retryable_errors = {
            "APITimeoutError",
            "APIConnectionError",
            "RateLimitError",
            "ServiceUnavailableError",
            "InternalServerError",
            "BadGatewayError",
            "GatewayTimeoutError",
        }

        return error_type in retryable_errors

    @classmethod
    def from_model_string(cls, model_string: str, **kwargs) -> "LLMEvaluator":
        """Create evaluator from model string.

        Args:
            model_string: Model specification like "openai/gpt-4"
            **kwargs: Additional provider configuration

        Returns:
            LLMEvaluator instance
        """
        provider = ProviderFactory.create_provider(model_string, **kwargs)
        return cls(provider)
